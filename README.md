# Uber AI Customer Support Agent & Evaluation Pipeline

An explainable, locally executable AI Support Assistant for **@Uber_Support**, built for the Hiver SDE Intern assignment. 

The pipeline processes customer tweets and outputs structured decisions:
```json
{
  "intent": "Lost_Item",
  "reply": "We understand losing an item is stressful. Please navigate to 'Activity' > Select your ride > 'Find lost item' in the Uber app...",
  "escalate": true,
  "reason": "Immediate live coordination required to recover personal property from driver."
}
```
⚡ Key Features & Constraints Adherence
1.Zero-API-Key Architecture: 100% open-source, locally executable execution using Scikit-Learn and HuggingFace pipelines.

2.Compliant Golden Dataset: 200 hand-labeled customer queries strictly stratified across four custom intents.

3.Multilevel Baselines: Evaluated against a Majority Class (Trivial) and TF-IDF + Logistic Regression (Simple) baseline.

4.Deterministic Escalation: Rule-based safety routing to guarantee 0% false negatives on emergency/safety incidents.

5.Automated Evaluation Harness: LLM-as-a-judge scoring heuristic evaluating politeness, grounding, and escalation accuracy.

## 1. Problem Framing
* **What "Good" means for @Uber_Support:** First triage that prioritizes passenger safety as well as lost property recovery through instant human escalation, while providing ground self-service paths for standard fare disputes.
* * **What We chose not to Build:** Real-time external API balance lookups and multi-turn thread memory. The agent focuses strictly on deterministic single-turn routing to keep latency under 200ms.

## 2. Results vs. Baselines
  * **Trivial Baseline (Majority Class):** Achieved **~38%** accuracy by predicting `Ride_Issue` for incoming message.
  * **Simple ML Baselines (TF-IDF + Logistic Regression):** Achieved **~73%** accuracy across four different intents (`Ride_Issue`, `Payment_Account`, `Lost_Item`, `General_Inquiry`).
  * **AI Agent Pipeline:** Enhanced simple classification by pairing predictions with intent-grounded historicalresponse templates and a deterministic rule-based safety escalation engine.

## 3. Failed Modes
1. **Sarcasm Detection:** *"Thanks Uber for getting me a dirty and unhygenic car!"* - Misclassified as General_Inquiry due to the positive tokens like "thanks".\
2. **Compound / Multi-Intent Inquiries:** *"Driver took a long detour and overcharged me"* - COntains both a route complaint and a billing dispute: the pipeline is forced to pick only one.
3. **Ambigious / Single-Word Tweets:** *"Help"* or *"DM"* - Lacks sufficient semantic tokens for the TF-IDF vectorization.
4. **Driver vs. Rider Role Ambiguity:** Driver complaints about earnings or rider ratings misclassified under passenger categories.
5. **Hyperbolic Urgency:** Users expressing extreme frustration over simple delays triggering emergency safety escalation.

## 4. What is Misleading About My Headline Number? 
An overall accuracy score of ~73% suggests the system is reasonably effective, but this metric is heavily driven by frequent, formulaic billing questions. In customer service, **false negatives on safety issues carry severe real-world consequences**. Classifying an emergency or harassment report as a standard ride inquiry is an unacceptable failure that standard aggregate accuracy masks completely.

## 5. What I Would Do Next With One More Week
* Implement a hybrid Bi-Encoder (Sentence-Transformers) semantic search across historical resolved tickets.
* Train a dedicated dual-head transformer: Head A for intent classification and Head B specifically for safety risk scoring.
* Introduce an active-learning feedback loop where human support agents can correct misrouted tickets directly in the UI.

## 6. Decision Log (10 Key Decisions)
1. **Brand Selection:** Chose `@Uber_Support` due to high contrast between routine fare queries and high-stakes safety incidents.
2. **Intent Scope:** Capped intent categories at four to maximize classifier precision on a 200-sample set.
3. **Deterministic Escalation:** Kept safety escalation rule-based rather than purely probabilistic to prevent critical safety escapes.
4. **Chunked CSV Parsing:** Implemented 150k row chunks to prevent memory crashes on Colab/local machines.
5. **Balanced Class Weighting:** Used `class_weight='balanced'` in Logistic Regression to prevent majority-class bias.
6. **Grounding Strategy:** Grounded replies in historical brand action links (e.g., directing riders to the in-app "Find Lost Item" flow) rather than generating unconstrained text.
7. **Rule-Based Evaluation Rubric:** Implemented transparent rule checks for politeness, calls-to-action, and escalation alignment.
8. **Statified Sampling:** Used stratified splitting to ensure rare classes like `Lost_Item` were represented in both training and test sets.
9. **Exclusion of Non-English Tweets:** Filtered foreign character sets to preserve English vectorization quality.
10. **Zero-API-Key Architecture:** Built the entire pipeline with open-source Scikit-Learn and Pandas to ensure reproducibility in under 15 minutes.

**Note: An AI assistant was used to help structure this report, help with the python piple syntax, and expedite the data processing. 
