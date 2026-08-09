# Temporal operators usable in `assert property`

Source: *IEEE Std 1800-2017, IEEE Standard for SystemVerilog—Unified Hardware Design, Specification, and Verification Language*, clauses 16.9.2, 16.9.3, 16.9.6, 16.9.8–16.9.10, 16.12.2, 16.12.7–16.12.13, and 16.12.15.

Scope: sequence/property operators with temporal or sampled-time semantics. Pure Boolean/combinational operators (for example, property/sequence `and`, `or`, and `not`) are intentionally excluded. In the examples, `clk` is the assertion clock and `a`, `b`, `c`, `req`, `ack`, and `data` are sampled signals. Each example is a complete concurrent-assertion statement.

## Sequence operators and sampled-value functions

| IEEE clause | Operator | Original syntax form | Natural-language explanation | Example usage |
|---|---|---|---|---|
| 16.9.2 | Consecutive repetition, exact | `sequence_expr [* constant_expression]` | Match the operand sequence exactly the specified number of times in succession. Successive matches begin one clock tick after the preceding match ends. | `assert property (@(posedge clk) a |-> b[*3]);` |
| 16.9.2 | Consecutive repetition, range | `sequence_expr [* min:max]` | Match the operand consecutively between `min` and `max` times. `$` may be the finite-but-unbounded upper limit. | `assert property (@(posedge clk) a |-> b[*2:4] ##1 c);` |
| 16.9.2 | Zero-or-more consecutive repetition | `sequence_expr [*]` | Match zero or more consecutive iterations; this is shorthand for `[*0:$]`. | `assert property (@(posedge clk) a |-> b[*] ##1 c);` |
| 16.9.2 | One-or-more consecutive repetition | `sequence_expr [+]` | Match one or more consecutive iterations; this is shorthand for `[*1:$]`. | `assert property (@(posedge clk) a |-> b[+] ##1 c);` |
| 16.9.2 | Goto repetition | `boolean_expr [-> min:max]` | Match the Boolean expression `min` through `max` times, not necessarily consecutively, with no extra match between counted matches; the repetition ends on the last counted match. | `assert property (@(posedge clk) req |-> ack[->2:4] ##1 c);` |
| 16.9.2 | Nonconsecutive repetition | `boolean_expr [= min:max]` | Count `min` through `max` nonconsecutive matches with no extra match between them. Unlike goto repetition, the sequence may finish after the final counted match, provided the operand remains false until the finish. | `assert property (@(posedge clk) req |-> ack[=2:4] ##1 c);` |
| 16.9.3 | Sampled value | `$sampled(expression)` | Return the value sampled for the current assertion time slot. This is most useful in an assertion action block; inside the property expression it is normally redundant. | `assert property (@(posedge clk) a == b) else $error("a=%b b=%b", $sampled(a), $sampled(b));` |
| 16.9.3 | Rose | `$rose(expression [, clocking_event])` | True when the least-significant bit is sampled as `1` now and was not `1` at the preceding sampling tick. | `assert property (@(posedge clk) $rose(req) |=> ack);` |
| 16.9.3 | Fell | `$fell(expression [, clocking_event])` | True when the least-significant bit is sampled as `0` now and was not `0` at the preceding sampling tick. | `assert property (@(posedge clk) $fell(req) |=> !ack);` |
| 16.9.3 | Stable | `$stable(expression [, clocking_event])` | True when the complete sampled value is unchanged from the preceding sampling tick. | `assert property (@(posedge clk) a |-> $stable(data));` |
| 16.9.3 | Changed | `$changed(expression [, clocking_event])` | True when the complete sampled value differs from the preceding sampling tick. | `assert property (@(posedge clk) a |-> $changed(data));` |
| 16.9.3 | Past | `$past(expr [, ticks [, gate [, clocking_event]]])` | Return the sampled value from `ticks` prior ticks of the relevant clock; when a gate is supplied, count only ticks on which that gate is true. The default depth is one. | `assert property (@(posedge clk) ack |-> $past(req, 2));` |
| 16.9.6 | Intersection | `sequence_expr1 intersect sequence_expr2` | Require both operand sequences to match from the same start and to finish on the same clock tick. | `assert property (@(posedge clk) a |-> ((b ##1 c) intersect (b[*2])));` |
| 16.9.8 | First match | `first_match(sequence_expr)` | Select only the chronologically earliest match of a sequence that may have several possible end points. | `assert property (@(posedge clk) a |-> first_match(##[1:4] b) ##1 c);` |
| 16.9.9 | Throughout | `expression throughout sequence_expr` | Require the expression to be true at every clock tick covered by a finite match of the sequence. | `assert property (@(posedge clk) req |-> (req throughout (1'b1 ##[1:4] ack)));` |
| 16.9.10 | Within | `sequence_expr1 within sequence_expr2` | Require a match of the first sequence to occur wholly inside the start-to-end interval of a match of the second sequence. | `assert property (@(posedge clk) a |-> ((b ##1 c) within (1'b1 ##[1:5] ack)));` |

Notes for 16.9.2: goto and nonconsecutive repetition accept a Boolean-expression operand, whereas consecutive repetition accepts a general sequence. A `$` upper bound denotes an unbounded set of possible **finite** matches; it does not create an infinite sequence match.

Notes for 16.9.3: `$rose` and `$fell` inspect only the least-significant bit. `$stable` and `$changed` compare the entire expression. At/before the first sampling event, change functions compare against the type's default sampled value. The optional `$past` gate is not an implication condition: it filters which clock ticks are counted.

## Property operators

| IEEE clause | Operator | Original syntax form | Natural-language explanation | Example usage |
|---|---|---|---|---|
| 16.12.2 | Strong sequence property | `strong(sequence_expr)` | The property is true only if the sequence has a nonempty match. An unbounded search must eventually reach a successful finite match. | `assert property (@(posedge clk) a |-> strong(##[1:$] b));` |
| 16.12.2 | Weak sequence property | `weak(sequence_expr)` | The property is true if the sequence has a match; it also remains true when an unbounded search is cut short because no later clock tick exists. | `assert property (@(posedge clk) a |-> weak(##[1:$] b));` |
| 16.12.7 | Overlapped implication | `sequence_expr |-> property_expr` | For every antecedent match, start the consequent on the antecedent's ending tick. If the antecedent has no match, the implication succeeds vacuously. | `assert property (@(posedge clk) req |-> ack);` |
| 16.12.7 | Nonoverlapped implication | `sequence_expr |=> property_expr` | For every antecedent match, start the consequent on the tick after the antecedent ends. If the antecedent has no match, the implication succeeds vacuously. | `assert property (@(posedge clk) req |=> ack);` |
| 16.12.8 | Property implication | `property_expr1 implies property_expr2` | Evaluate both properties from the same starting tick; if the first property succeeds, the second must succeed. | `assert property (@(posedge clk) (always req) implies (always ack));` |
| 16.12.8 | Property equivalence | `property_expr1 iff property_expr2` | Evaluate both properties from the same starting tick and require both implications, so the two properties have the same truth outcome. | `assert property (@(posedge clk) (always req) iff (always ack));` |
| 16.12.9 | Overlapped followed-by | `sequence_expr #-# property_expr` | Concatenate a sequence with a property starting on the sequence's ending tick. Unlike implication, it does not succeed vacuously merely because the sequence has no match. | `assert property (@(posedge clk) (req ##1 ack) #-# always c);` |
| 16.12.9 | Nonoverlapped followed-by | `sequence_expr #=# property_expr` | Concatenate a sequence with a property starting one tick after the sequence ends. It is a concatenation-style operator, not vacuous implication. | `assert property (@(posedge clk) (req ##1 ack) #=# always c);` |
| 16.12.10 | Weak nexttime | `nexttime property_expr` | If a next tick exists, the property must hold there; if no next tick exists, the weak form is still true. | `assert property (@(posedge clk) a |-> nexttime b);` |
| 16.12.10 | Indexed weak nexttime | `nexttime[n] property_expr` | Apply weak-nexttime semantics at the `n`th future tick. It succeeds weakly if the clock stops before that tick. | `assert property (@(posedge clk) a |-> nexttime[2] b);` |
| 16.12.10 | Strong nexttime | `s_nexttime property_expr` | Require a next clock tick to exist and require the property to hold at that tick. | `assert property (@(posedge clk) a |-> s_nexttime b);` |
| 16.12.10 | Indexed strong nexttime | `s_nexttime[n] property_expr` | Require at least `n` future ticks and require the property at the `n`th future tick. | `assert property (@(posedge clk) a |-> s_nexttime[2] b);` |
| 16.12.11 | Weak always | `always property_expr` | Require the property at every current and future clock tick; the operator does not itself require more ticks to exist. | `assert property (@(posedge clk) a |-> always b);` |
| 16.12.11 | Ranged weak always | `always[min:max] property_expr` | Require the property at every existing tick in the specified future range. Missing ticks do not cause failure, and `$` is allowed as the upper bound. | `assert property (@(posedge clk) a |-> always[2:5] b);` |
| 16.12.11 | Ranged strong always | `s_always[min:max] property_expr` | Require every tick in the bounded range to exist and require the property at each of those ticks. An unbounded upper limit is illegal. | `assert property (@(posedge clk) a |-> s_always[2:5] b);` |
| 16.12.12 | Weak nonoverlapping until | `property_expr1 until property_expr2` | Require the first property through every tick before the second becomes true. The second need never occur; if it occurs now, the first need not hold now. | `assert property (@(posedge clk) req until ack);` |
| 16.12.12 | Strong nonoverlapping until | `property_expr1 s_until property_expr2` | As above, but require the second property to become true at some current or future tick. | `assert property (@(posedge clk) req s_until ack);` |
| 16.12.12 | Weak overlapping until | `property_expr1 until_with property_expr2` | Require the first property through and including a tick where the second holds; the second need never occur in the weak form. | `assert property (@(posedge clk) req until_with ack);` |
| 16.12.12 | Strong overlapping until | `property_expr1 s_until_with property_expr2` | Require the second property eventually and require the first through and including the terminating tick. | `assert property (@(posedge clk) req s_until_with ack);` |
| 16.12.13 | Strong eventually | `s_eventually property_expr` | Require the property to become true at some current or future clock tick. | `assert property (@(posedge clk) req |-> s_eventually ack);` |
| 16.12.13 | Ranged weak eventually | `eventually[min:max] property_expr` | Require a match in the bounded range if every tick in that range exists; if the clock ends before the range is fully available, the weak form is true. `$` is illegal here. | `assert property (@(posedge clk) req |-> eventually[2:5] ack);` |
| 16.12.13 | Ranged strong eventually | `s_eventually[min:max] property_expr` | Require the property at some tick in the range. The range may be unbounded with `$`. | `assert property (@(posedge clk) req |-> s_eventually[2:$] ack);` |

## Strong/weak summary (16.12.15)

| Strength | Operators | Termination/clock requirement |
|---|---|---|
| Strong | `strong`, `s_nexttime`, `s_always`, `s_eventually`, `s_until`, `s_until_with` | Require the relevant future terminating condition and enough clock ticks for it to occur. |
| Weak | `weak`, `nexttime`, `always`, `eventually`, `until`, `until_with` | Do not independently require the terminating condition or additional clock ticks. |

`|->`/`|=>` are not classified as weak operators by 16.12.15. Their special behavior is vacuity: an implication attempt succeeds when its antecedent has no match. `#-#`/`#=#` do not have that implication vacuity.

## JasperGold syntax-validation status

All 38 table examples were placed in one self-contained module and checked with **Cadence JasperGold 2026.03p001 (64-bit)** after loading `/home/shared/modules/cadence/IC231`. The validation flow followed the repository harness: `clear -all`, `analyze -clear`, `analyze -sv12`, and `elaborate`.

Validation result: **PASS** (Jasper exit status 0; no syntax or elaboration errors). Jasper emitted only informational/semantic diagnostics: one warning for the intentionally weak unbounded sequence in the `weak(##[1:$] b)` example and informational notices that two until examples contain liveness safety components. No operator was excluded for lack of JasperGold syntax support.
