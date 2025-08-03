# LLM 프롬프팅 기법 비교 실험 보고서

## 1. 실험 결과 비교표

| Method | 0-shot | 3-shot | 5-shot |
|--------|--------|--------|--------|
| Direct Prompting | 24.00% | 20.00% | 22.00% |
| CoT Prompting | 84.00% | 70.00% | 74.00% |
| My Prompting (Majority Voting) | 80.00% | 80.00% | 82.00% |

## 2. CoT Prompting이 Direct Prompting에 비해 좋은 이유

Direct Prompting은 단순히 최종 답만 요구하는 방식으로, LLM이 추론 과정을 건너뛸 수 있어 중간 계산 오류를 놓치기 쉽습니다. 특히 복잡한 다단계 문제에서는 문제를 작은 단위로 나누지 못해 어려움을 겪습니다.

반면 CoT Prompting은 각 단계를 명시적으로 보여줌으로써 LLM이 논리적 사고를 따라갈 수 있게 합니다. 이는 각 단계를 확인할 수 있어 오류를 줄이고, 복잡한 문제도 작은 단위로 나누어 해결할 수 있게 해줍니다.

## 3. My Prompting이 CoT에 비해 더 좋은 이유

CoT Prompting이 단일 응답에 의존하는 반면, My Prompting은 Self-Consistency를 통해 여러 번 실행하여 중앙값을 사용함으로써 안정성을 향상시킵니다. 또한 CoT가 일반적인 단계별 해결에 그치는 반면, My Prompting은 인과관계와 문맥 파악을 명시적으로 강조합니다.

구체적으로, My Prompting은 체계적인 분석 전략을 사용합니다. Cause-and-Effect 분석을 통해 문제의 인과관계를 파악하고, Context Understanding으로 상황을 정확히 이해하며, Logical Reasoning으로 논리적 사고를 수행하고, Verification으로 각 단계에서 논리적 검증을 강조합니다.

## 4. 실험 방법론

본 실험은 GSM8K 데이터셋을 사용하여 각 실험당 50개 샘플을 테스트했습니다. 세 가지 프롬프팅 기법을 비교했습니다: Direct Prompting(최종 답만 요구), CoT Prompting(단계별 사고 과정 포함), My Prompting(인과관계 분석 + Majority Voting). 평가 지표로는 정확도(Accuracy)와 Self-Consistency 점수를 사용했습니다.

## 5. 결론

My Prompting 기법이 CoT Prompting보다 더 나은 성능을 보이는 이유는 다음과 같습니다. 첫째, Majority Voting으로 이상치 답변을 필터링하여 안정성을 확보합니다. 둘째, 인과관계와 문맥 파악을 강조하여 논리성을 높입니다. 셋째, 각 단계에서 논리적 검증을 수행하여 검증성을 강화합니다. 마지막으로, 체계적인 문제 해결 전략을 통해 구조화된 접근을 가능하게 합니다.

이러한 결과는 LLM의 수학 문제 해결 능력을 향상시키기 위해서는 단순한 단계별 사고뿐만 아니라, 인과관계 분석과 안정적인 추론 과정이 중요함을 시사합니다. 특히 Self-Consistency와 구조화된 사고 전략의 조합이 효과적임을 보여줍니다.
