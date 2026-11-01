// DBSP external probe for the declared BiWorldDB workload.
//
// The workload event stream is emitted by emit_events.cpp from the same C++ generator as the
// authored arms. Events are fed to an actual DBSP (0.369) circuit as a Z-set input stream; an
// apply_mut operator maintains the authored per-group Fenwick prefix and emits exact answers
// as a Z-set. Every phase is charged.
use std::env;
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use dbsp::typed_batch::IndexedZSetReader;
use dbsp::operator::Inspect;
use dbsp::algebra::ZWeight;
use dbsp::typed_batch::FallbackZSet as TypedZSet;
use dbsp::circuit::Runtime;
use dbsp::utils::tuple::{Tup2, Tup4};

#[derive(Clone, Debug, Default)]
struct Ev {
    seq: u64,
    kind: u32,
    state: u32,
    qidx: u32,
    g: i32,
    w: i32,
    a: i64,
    b: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Ans {
    state: u32,
    qidx: u32,
    ans: i64,
}

struct State {
    times: Vec<Vec<i64>>,
    bit: Vec<Vec<i64>>,
    mul: Vec<i64>,
}

impl State {
    fn add(&mut self, g: usize, pos: usize, v: i64) {
        let n = self.times[g].len();
        let bit = &mut self.bit[g];
        let mut i = pos + 1;
        while i <= n {
            bit[i] += v;
            i += i & (!i + 1);
        }
    }
    fn prefix(&self, g: usize, k: usize) -> i64 {
        let bit = &self.bit[g];
        let mut s = 0i64;
        let mut i = k;
        while i > 0 {
            s += bit[i];
            i -= i & (!i + 1);
        }
        s
    }
    fn apply(&mut self, ev: &Ev, out: &mut Vec<Ans>) {
        match ev.kind {
            0 => {
                let g = ev.g as usize;
                let pos = self.times[g].binary_search(&ev.a).expect("known coordinate") as usize;
                self.add(g, pos, i64::from(ev.w) * ev.b);
            }
            1 => self.mul[ev.g as usize] = ev.a,
            2 => {
                let mut ans = 0i64;
                for g in 0..self.times.len() {
                    let lo = self.times[g].partition_point(|&x| x < ev.a);
                    let hi = self.times[g].partition_point(|&x| x < ev.b);
                    ans += self.mul[g] * (self.prefix(g, hi) - self.prefix(g, lo));
                }
                out.push(Ans { state: ev.state, qidx: ev.qidx, ans });
            }
            _ => unreachable!(),
        }
    }
}

fn encode(ev: &Ev) -> Tup4<u64, u64, i64, i64> {
    let packed = u64::from(ev.kind)
        | (u64::from(ev.state) << 2)
        | (u64::from(ev.qidx) << 14)
        | ((ev.g as u64 & 0x1FF) << 20)
        | (((ev.w < 0) as u64) << 29);
    Tup4(ev.seq, packed, ev.a, ev.b)
}

fn decode(t: &Tup4<u64, u64, i64, i64>) -> Ev {
    let packed = t.1;
    Ev {
        seq: t.0,
        kind: (packed & 0x3) as u32,
        state: ((packed >> 2) & 0xFFF) as u32,
        qidx: ((packed >> 14) & 0x3F) as u32,
        g: ((packed >> 20) & 0x1FF) as i32,
        w: if (packed >> 29) & 1 == 1 { -1 } else { 1 },
        a: t.2,
        b: t.3,
    }
}

fn read_u32(data: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
    *off += 4;
    v
}
fn read_i32(data: &[u8], off: &mut usize) -> i32 {
    let v = i32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
    *off += 4;
    v
}
fn read_i64(data: &[u8], off: &mut usize) -> i64 {
    let v = i64::from_le_bytes(data[*off..*off + 8].try_into().unwrap());
    *off += 8;
    v
}

fn load_fixture(path: &str) -> (Vec<i64>, Vec<i64>, Vec<i32>, usize, usize) {
    let mut data = Vec::new();
    File::open(path).unwrap().read_to_end(&mut data).unwrap();
    assert_eq!(&data[0..8], b"BWBASE02");
    let mut off = 8usize;
    let n = read_i64(&data, &mut off) as usize;
    let groups = read_i32(&data, &mut off) as usize;
    let _unit = read_i32(&data, &mut off);
    let _lines = read_i64(&data, &mut off);
    let _missing = read_i64(&data, &mut off);
    let _tmin = read_i64(&data, &mut off);
    let _tmax = read_i64(&data, &mut off);
    let _scale = read_i64(&data, &mut off);
    assert_eq!(off, 64);
    let mut t = Vec::with_capacity(n);
    let mut amount = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    for _ in 0..n {
        t.push(read_i64(&data, &mut off));
        amount.push(read_i64(&data, &mut off));
        g.push(read_i32(&data, &mut off));
        let _pad = read_i32(&data, &mut off);
    }
    let mut mul = Vec::with_capacity(groups);
    for _ in 0..groups {
        mul.push(read_i64(&data, &mut off));
    }
    (t, amount, g, groups, n)
}

fn load_events(path: &str) -> Vec<Ev> {
    let mut data = Vec::new();
    File::open(path).unwrap().read_to_end(&mut data).unwrap();
    assert_eq!(&data[0..8], b"BWLEV001");
    let mut off = 8usize;
    let n = read_i64(&data, &mut off) as u64;
    let mut events = Vec::with_capacity(n as usize);
    for seq in 0..n {
        let kind = read_u32(&data, &mut off);
        let state = read_u32(&data, &mut off);
        let qidx = read_u32(&data, &mut off);
        let g = read_i32(&data, &mut off);
        let w = read_i32(&data, &mut off);
        let a = read_i64(&data, &mut off);
        let b = read_i64(&data, &mut off);
        events.push(Ev { seq, kind, state, qidx, g, w, a, b });
    }
    assert_eq!(off, data.len());
    events
}

fn main() {
    let mut fixture = String::new();
    let mut events_path = String::new();
    let mut out = String::new();
    let mut oracle = String::new();
    let mut q = 2usize;
    let mut batch = 64usize;
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" => { fixture = args[i + 1].clone(); i += 2; }
            "--events" => { events_path = args[i + 1].clone(); i += 2; }
            "--out" => { out = args[i + 1].clone(); i += 2; }
            "--oracle" => { oracle = args[i + 1].clone(); i += 2; }
            "--q" => { q = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch = args[i + 1].parse().unwrap(); i += 2; }
            _ => { eprintln!("unknown arg {}", args[i]); std::process::exit(2); }
        }
    }
    let t0 = Instant::now();
    let (t, amount, g, groups, n) = load_fixture(&fixture);
    let mut events = load_events(&events_path);
    events.sort_by_key(|e| e.seq);
    let t_load = t0.elapsed().as_nanos() as u64;

    let mut times: Vec<Vec<i64>> = vec![Vec::new(); groups];
    for i in 0..n { times[g[i] as usize].push(t[i]); }
    for ev in events.iter().filter(|e| e.kind == 0) { times[ev.g as usize].push(ev.a); }
    for gv in times.iter_mut() { gv.sort_unstable(); gv.dedup(); }
    let mut bit: Vec<Vec<i64>> = times.iter().map(|v| vec![0i64; v.len() + 1]).collect();
    for i in 0..n {
        let gv = g[i] as usize;
        let pos = times[gv].binary_search(&t[i]).unwrap();
        let mut j = pos + 1;
        while j <= times[gv].len() { bit[gv][j] += amount[i]; j += j & (!j + 1); }
    }
    let mul = vec![1000i64; groups];
    let state: Arc<Mutex<State>> = Arc::new(Mutex::new(State { times, bit, mul }));
    let t_setup = t0.elapsed().as_nanos() as u64;

    let t1 = Instant::now();
    let answers_log: Arc<Mutex<Vec<Ans>>> = Arc::new(Mutex::new(Vec::new()));
    let answers_in = Arc::clone(&answers_log);
    let (mut dbsp, (mut input,)) = Runtime::init_circuit(1, move |circuit| {
        let (stream, handle) = circuit.add_input_zset::<Tup4<u64, u64, i64, i64>>();
        let state = Arc::clone(&state);
        let answers_in = Arc::clone(&answers_in);
        let out = stream.apply_mut_named("ivm_prefix", move |zset: &TypedZSet<Tup4<u64, u64, i64, i64>>| {
            let mut evs: Vec<Ev> = zset.iter().map(|(k, _v, _w)| decode(&k)).collect();
            evs.sort_by_key(|e| e.seq);
            let mut answers: Vec<Ans> = Vec::new();
            {
                let mut st = state.lock().unwrap();
                for ev in evs { st.apply(&ev, &mut answers); }
            }
            answers_in.lock().unwrap().extend(answers.iter().cloned());
            let tuples: Vec<Tup2<Tup2<u64, i64>, ZWeight>> = answers
                .into_iter()
                .map(|a| Tup2(Tup2((u64::from(a.state) << 6) | u64::from(a.qidx), a.ans), 1 as ZWeight))
                .collect();
            TypedZSet::from_keys((), tuples)
        });
        out.inspect(|_z: &TypedZSet<Tup2<u64, i64>>| {});
        Ok((handle,))
    }).unwrap();
    for chunk in events.chunks(batch) {
        for ev in chunk { input.push(encode(ev), 1i64); }
        dbsp.transaction().unwrap();
    }
    let t_exec = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();

    dbsp.kill().expect("dbsp kill");
    let mut collected: Vec<Ans> = answers_log.lock().unwrap().clone();
    collected.sort_by_key(|a| (a.state, a.qidx));
    let mut out_vec = vec![0i64; 2048 * q];
    for a in &collected {
        if a.state == 0 { continue; }
        out_vec[((a.state as usize - 1) * q) + a.qidx as usize] = a.ans;
    }
    let t_extract = t2.elapsed().as_nanos() as u64;
    let mut ok: Option<bool> = None;
    let mut detail = String::new();
    if !oracle.is_empty() {
        let mut exp = Vec::new();
        File::open(&oracle).unwrap().read_to_end(&mut exp).unwrap();
        let exp2: Vec<i64> = exp.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        if exp2.len() != out_vec.len() {
            ok = Some(false);
            detail = format!("oracle length {} != {}", exp2.len(), out_vec.len());
        } else {
            let bad = out_vec.iter().zip(exp2.iter()).filter(|(a, b)| a != b).count();
            ok = Some(bad == 0);
            detail = if bad == 0 { "exact match".to_string() } else { format!("{bad} mismatches") };
        }
    }
    if !out.is_empty() {
        let mut bytes = Vec::with_capacity(out_vec.len() * 8);
        for v in &out_vec { bytes.extend_from_slice(&v.to_le_bytes()); }
        std::fs::write(&out, bytes).unwrap();
    }
    let rec = format!(
        "{{\"engine\":\"dbsp\",\"dbsp_version\":\"0.369.0\",\"events\":{},\"batch\":{},\"answers\":{},\"oracle_ok\":{},\"oracle_detail\":\"{}\",\"phases_ns\":{{\"load\":{},\"setup\":{},\"execute\":{},\"extract\":{}}},\"full_ns\":{}}}",
        events.len(), batch, out_vec.len(),
        ok.map(|x| x.to_string()).unwrap_or_else(|| "null".to_string()),
        detail, t_load, t_setup - t_load, t_exec, t_extract,
        t_setup + t_exec + t_extract
    );
    println!("{}", rec);
}
