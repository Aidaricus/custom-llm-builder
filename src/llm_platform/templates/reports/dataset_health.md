# Dataset Health Report

## 1. Sanitization Overview
* **Total Raw Pairs Processed:** {total_processed}
* **Pairs Passed:** {passed}
* **Rejection Rate:** {rejection_rate}%
  * Rejected due to bad structure: {rejected_structure}
  * Rejected due to length: {rejected_length}
  * Rejected due to AI Refusal: {rejected_refusal}
  * Rejected due to Duplication: {rejected_duplicate}

## 2. Token & Length Statistics (Passed SFT Data)
* **Total Valid Pairs:** {total_pairs}
* **Average Answer Length (Words):** {avg_words}
* **Max Answer Length:** {max_words}
* **Min Answer Length:** {min_words}

## 3. Spot-Checking (LLM-as-a-Judge)
* **Samples Evaluated (5%):** {sample_size}
* **Factual Accuracy Rate:** {factual_rate}%
* **Status:** {status_icon}

*Generated automatically by LLM Data Foundry.*