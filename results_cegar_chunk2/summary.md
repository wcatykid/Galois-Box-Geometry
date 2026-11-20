# Chunk 2: Full-Scale Evaluation Summary

All 2538 verb IS-A constraints used (no sub-sampling).

## Winners

- Best method on raw fix rate: Coordinate descent (step_size=0.3,iterations=20)
- Best method on net improvement: Coordinate descent (step_size=0.3,iterations=20)
- Best method on geometry preservation: No refinement (-)
- Best overall tradeoff: Coordinate descent (step_size=0.3,iterations=20)

## Head-to-Head

- Does coordinate descent beat hybrid on aggregate score? Yes
- Does hybrid produce fewer new violations? No
- Does hybrid have less box displacement? Yes

## Key Numbers (Best Overall: Coordinate descent)

- Fix rate: 100.0%
- New violations: 0
- Net improvement: 1114
- Final violation rate: 0.0%
- Galois gap change: +0.210490
- Mean box displacement: 0.255098
- Aggregate score: +1.7110

## Verdict

STRONG: recommended for production use

## Recommended Configuration

Method: Coordinate descent
Hyperparameters: step_size=0.3,iterations=20
