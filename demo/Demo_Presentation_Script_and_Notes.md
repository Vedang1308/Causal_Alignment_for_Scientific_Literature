# CCST Project DEMO Presentation: 12-Minute Speaker Script (ASU Official Deck)

**Course:** Software & Web Mining (SWM) — Fall 2026, Arizona State University  
**Group & Project:** Group 16 • Project 16 — Context-Conditioned Causal Alignment for Scientific Literature (CCST)  
**Submission Slides:** `Group16-Project16-FA26-Group-DEMO-Presentation.pptx` & `.pdf`  
**Live Demo App:** Open `code/ccst_interactive_demo.html` in Chrome/Safari  
**GitHub Repository:** https://github.com/Vedang1308/Causal_Alignment_for_Scientific_Literature  

---

## Presenter Assignments & Timing (Total: 12 Minutes)

| Slide # | Topic & Rubric Section | Speaker | Time |
| :---: | :--- | :--- | :---: |
| **1** | Cover Page (Group 16, Project 16, Members) | **Shravan Krishna Vijay** | 0:45 |
| **2** | Problem Definition [1 Pt] | **Shravan Krishna Vijay** | 1:30 |
| **3** | System Architecture: 4 Simple Steps [1 Pt] | **Sri Harshith Goli** | 1:00 |
| **4** | Core Algorithms: Dual-View & Context Gate [1 Pt] | **Sri Harshith Goli** | 1:15 |
| **5** | Datasets: 2,054 Real Papers & 240-Pair Benchmark [1 Pt] | **Subhash Shivaprasad** | 1:00 |
| **6** | Evaluations: Comparing Against Existing AI [2.5 Pts] | **Subhash Shivaprasad** | 1:30 |
| **7** | Evaluations: Why Every Component Matters [2.5 Pts] | **Tarun Kumar Bantupalli** | 1:00 |
| **8** | UI Demo (1/2): Intuitive Claim Comparison [2.5 Pts] | **Vedang Vasant Avaghade** | 1:30 |
| **9** | UI Demo (2/2): Standard AI Toggle & Proof [2.5 Pts] | **Vedang Vasant Avaghade** | 1:15 |
| **10** | Project Plan & Division of Work [1 Pt] | **Tarun Kumar Bantupalli** | 0:45 |
| **11** | Thank You, Repository Layout & GitHub Link | **Vedang Vasant Avaghade** | 0:30 |

---

## How to Run the 2-Minute Live Demo Smoothly (Slides 8–9)

Keep `code/ccst_interactive_demo.html` open in a browser tab right beside your slides:

1. **Start on Tab 1 (`Metformin & Blood Sugar`):**
   - Point to the two simple cards: *Paper A studied isolated mouse liver cells (10 mM)*, while *Paper B studied human diabetic patients (500 mg)*.
   - Point out our Verdict Box at the bottom: **NOT A CONTRADICTION — Different Experiment Conditions**.
2. **Do the "Aha!" Moment Click (1 Click):**
   - Click the red toggle button **`✗ Standard AI (Ignores Context)`**.
   - Watch the card immediately turn bright red: **`✗ FALSE CONTRADICTION ALERT (91.5%)`**.
   - Say: *"This shows live why Sentence-BERT and standard LLMs fail 40%–44% of the time—when context isn't checked, mouse lab results look like contradictions with human trials."*
   - Switch back to **`✓ Our System (Checks Context)`** to show how our Context Gate blocks the false alarm.
3. **Click Tab 3 (`Security vs. Viruses`):**
   - Say: *"Now look at cross-domain discovery: Computer Security papers share almost zero vocabulary with Virology, yet our graph matcher aligns the tipping-point cascade (`Faults > Threshold -> Failure`) with `Unvaccinated > Threshold -> Outbreak` at 92.4% confidence."*
4. **Click `+ Show Technical Proof & Disagreement Certificate`:**
   - Briefly show the calibrated 5-way confidence bars and JSON certificate grounded to paper IDs and character spans.
5. *(Optional if asked)* Click **`+ Try Custom Claims`** and click **`Compare Claims Now`** to analyze user-typed sentences live!
