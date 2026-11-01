// cross_family_weights.hpp — generated from results/cross_family_planner_20261029/weights_v6_structured.json
// Structured analytic cost model (standardized ridge lambda=1, multi-scale training).
#pragma once
namespace bwh {
inline constexpr int kCrossFamilyHybridDims = 6;
inline constexpr int kCrossFamilyClassicDims = 5;
inline constexpr double kCrossFamilyHybrid[kCrossFamilyHybridDims] = {3.36499629487, 0.543646882876, 0.445102074652, -0.0749876232599, -0.0977809037872, 0.822234704484};
inline constexpr double kCrossFamilyHybridMu[kCrossFamilyHybridDims] = {0, 6.43012784483, 0.0372459243103, 0.0756090212069, 0.06371091897, 0.653022413793};
inline constexpr double kCrossFamilyHybridSd[kCrossFamilyHybridDims] = {1, 9.99505444665, 0.0614852413007, 0.219345089599, 0.142901286317, 1.18042812356};
inline constexpr double kCrossFamilyClassic[kCrossFamilyClassicDims] = {4.07243172479, 1.09426222271, 0.484636440998, 0.45747724185, -0.0817634754009};
inline constexpr double kCrossFamilyClassicMu[kCrossFamilyClassicDims] = {0, 0.00813608568515, 0.0654679972084, 24.9262762416, 0.5334875};
inline constexpr double kCrossFamilyClassicSd[kCrossFamilyClassicDims] = {1, 0.0140221797546, 0.128775633787, 142.996826655, 0.941248533526};
}  // namespace bwh
