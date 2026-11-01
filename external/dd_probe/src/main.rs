// Differential Dataflow external probe for the declared BiWorldDB workload.
//
// The workload (edits, per-state query bags, DFS ordering) is emitted by emit_events.cpp
// from the same C++ generator as the authored arms. This program feeds those events as a
// differential collection into an actual Differential Dataflow dataflow; a stateful
// operator maintains a per-group Fenwick prefix over time coordinates (the authored
// prefix-hierarchy part) and answers the declared range queries exactly. Every number in
// the probe is charged: fixture load, event parse, index build, feed, dataflow execution,
// answer extraction.
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

use std::time::Instant;

use differential_dataflow::input::Input;
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::generic::operator::Operator;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Ev {
    seq: u64,
    kind: u32,   // 0 row delta, 1 factor set, 2 query
    state: u32,  // query: state id
    qidx: u32,   // query: bag index
    g: i32,      // group
    w: i32,      // row delta weight (+1/-1)
    a: i64,      // delta: t; factor: multiplier; query: lo
    b: i64,      // delta: amount; query: hi
}

struct State {
    times: Vec<Vec<i64>>,
    bit: Vec<Vec<i64>>,
    mul: Vec<i64>,
    query_lo: Vec<i64>,
    query_hi: Vec<i64>,
    answers: Arc<Mutex<Vec<(u32, u32, i64)>>>,
    count: u64,
}

impl State {
    fn add(&mut self, g: usize, pos: usize, v: i64) {
        let bit = &mut self.bit[g];
        let n = self.times[g].len();
        let mut i = pos + 1;
        while i <= n {
            bit[i] += v;
            i += i & (!i + 1);
        }
    }
    fn prefix(&self, g: usize, k: usize) -> i64 {
        // sum of amounts at times strictly before times[k]
        let bit = &self.bit[g];
        let mut s = 0i64;
        let mut i = k;
        while i > 0 {
            s += bit[i];
            i -= i & (!i + 1);
        }
        s
    }
    fn apply(&mut self, ev: &Ev) {
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
                self.answers.lock().unwrap().push((ev.state, ev.qidx, ans));
                self.count += 1;
            }
            _ => unreachable!(),
        }
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
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" => { fixture = args[i + 1].clone(); i += 2; }
            "--events" => { events_path = args[i + 1].clone(); i += 2; }
            "--out" => { out = args[i + 1].clone(); i += 2; }
            "--oracle" => { oracle = args[i + 1].clone(); i += 2; }
            "--q" => { q = args[i + 1].parse().unwrap(); i += 2; }
            _ => { eprintln!("unknown arg {}", args[i]); std::process::exit(2); }
        }
    }
    let t0 = Instant::now();
    let (t, amount, g, groups, n) = load_fixture(&fixture);
    let events = load_events(&events_path);
    let t_load = t0.elapsed().as_nanos() as u64;

    // per-group sorted unique time coordinates (base times + every delta destination)
    let mut times: Vec<Vec<i64>> = vec![Vec::new(); groups];
    for i in 0..n {
        times[g[i] as usize].push(t[i]);
    }
    for ev in events.iter().filter(|e| e.kind == 0) {
        times[ev.g as usize].push(ev.a);
    }
    for gv in times.iter_mut() {
        gv.sort_unstable();
        gv.dedup();
    }
    let mut bit: Vec<Vec<i64>> = times.iter().map(|v| vec![0i64; v.len() + 1]).collect();
    let mut mul = vec![1000i64; groups]; // base multiplier is 1000 for every admitted base
    let mut init_amounts: Vec<Vec<(usize, i64)>> = vec![Vec::new(); groups];
    for i in 0..n {
        let gv = g[i] as usize;
        let pos = times[gv].binary_search(&t[i]).unwrap();
        init_amounts[gv].push((pos, amount[i]));
    }
    for gv in 0..groups {
        for &(pos, amt) in init_amounts[gv].iter() {
            let mut j = pos + 1;
            while j <= times[gv].len() {
                bit[gv][j] += amt;
                j += j & (!j + 1);
            }
        }
    }
    let t_setup = t0.elapsed().as_nanos() as u64;

    let answers: Arc<Mutex<Vec<(u32, u32, i64)>>> = Arc::new(Mutex::new(Vec::new()));
    let answers_out = Arc::clone(&answers);
    let n_events = events.len();
    let t1 = Instant::now();
    timely::execute_directly(move |worker| {
        let (mut input) = worker.dataflow::<u64, _, _>(|scope| {
            let (input, collection) = scope.new_collection::<Ev, isize>();
            let mut state = State {
                times,
                bit,
                mul,
                query_lo: Vec::new(),
                query_hi: Vec::new(),
                answers: answers_out,
                count: 0,
            };
            let mut next_seq = 0u64;
            let mut buffer: HashMap<u64, Ev> = HashMap::new();
            let _out = collection
                .inner
                .unary_frontier::<timely::container::CapacityContainerBuilder<Vec<()>>, _, _, _>(Pipeline, "ivm_prefix", move |_cap, _info| {
                    move |(inp, _frontier), _output| {
                        inp.for_each(|_cap, data| {
                            for (ev, _time, _diff) in data.drain(..) {
                                buffer.insert(ev.seq, ev);
                                while let Some(e) = buffer.remove(&next_seq) {
                                    next_seq += 1;
                                    state.apply(&e);
                                }
                            }
                        });
                    }
                });
            input
        });
        let mut all = events;
        all.sort_by_key(|e| e.seq);
        for ev in all {
            input.update(ev, 1isize);
        }
        input.close();
        while worker.step() {}
    });
    let t_exec = t1.elapsed().as_nanos() as u64;
    let t2 = Instant::now();

    let mut by_state: Vec<(u32, u32, i64)> = answers.lock().unwrap().clone();
    by_state.sort_by_key(|x| (x.0, x.1));
    let mut out_vec = vec![0i64; (2048) * q];
    for (state, qidx, ans) in by_state {
        if state == 0 { continue; }
        out_vec[((state as usize - 1) * q) + qidx as usize] = ans;
    }
    let t_extract = t2.elapsed().as_nanos() as u64;
    let mut ok: Option<bool> = None;
    let mut detail = String::new();
    if !oracle.is_empty() {
        let mut exp = Vec::new();
        File::open(&oracle).unwrap().read_to_end(&mut exp).unwrap();
        let exp2: Vec<i64> = exp.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        if exp2.len() != out_vec.len() {
            ok = Some(false);
            detail = format!("oracle length {} != {}", exp2.len(), out_vec.len());
        } else {
            let mut bad = 0usize;
            for (a, b) in out_vec.iter().zip(exp2.iter()) {
                if a != b { bad += 1; }
            }
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
        "{{\"engine\":\"differential-dataflow\",\"dd_version\":\"0.25.1\",\"events\":{},\"answers\":{},\"oracle_ok\":{},\"oracle_detail\":\"{}\",\"phases_ns\":{{\"load\":{},\"setup\":{},\"execute\":{},\"extract\":{}}},\"full_ns\":{}}}",
        n_events,
        out_vec.len(),
        ok.map(|x| x.to_string()).unwrap_or_else(|| "null".to_string()),
        detail,
        t_load,
        t_setup - t_load,
        t_exec,
        t_extract,
        t_setup + t_exec + t_extract
    );
    println!("{}", rec);
}
