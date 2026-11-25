# CCST Demo Presentation Script (Simple, Natural & Easy to Present)

**Group 16 • Project 16:** *Context-Conditioned Causal Alignment for Scientific Literature (CCST)*  
**Course:** Software & Web Mining (SWM) — Fall 2026, Arizona State University  
**Slides:** `demo/Group16-Project16-FA26-Group-DEMO-Presentation.pptx` (and `.pdf`)  
**Live Web Demo:** `demo/ccst_interactive_demo.html` (or `code/ccst_interactive_demo.html`)  
**GitHub Repo:** https://github.com/Vedang1308/Causal_Alignment_for_Scientific_Literature  

> **Tip for Presenters:** Don't memorize this word-for-word—read it once or twice so it feels conversational, speak at a relaxed pace (~12 minutes total), and make natural eye contact while pointing at the slides or clicking through the live demo.

---

## Speaker Split & Pacing (Total: ~12 Minutes)

| Slide | Topic | Speaker | Target Time |
| :---: | :--- | :--- | :---: |
| **1** | Intro & Team | **Shravan Krishna Vijay** | 0:45 |
| **2** | The Problem We're Solving | **Shravan Krishna Vijay** | 1:30 |
| **3** | How Our System Works (4 Steps) | **Sri Harshith Goli** | 1:00 |
| **4** | Core Ideas (Graph Matching & Context Gate) | **Sri Harshith Goli** | 1:15 |
| **5** | Datasets & Cleaning | **Subhash Shivaprasad** | 1:00 |
| **6** | Evaluation Results vs. Other AI | **Subhash Shivaprasad** | 1:30 |
| **7** | Ablations: Why Every Piece Matters | **Tarun Kumar Bantupalli** | 1:00 |
| **8** | Live UI Demo (Part 1: Comparing Papers) | **Vedang Vasant Avaghade** | 1:30 |
| **9** | Live UI Demo (Part 2: Standard AI Toggle & Proof) | **Vedang Vasant Avaghade** | 1:15 |
| **10** | Project Plan & Division of Work | **Tarun Kumar Bantupalli** | 0:45 |
| **11** | GitHub Repo & Wrap-Up | **Vedang Vasant Avaghade** | 0:30 |

---

## Spoken Script (Slide by Slide)

### Slide 1 — Cover Page *(Shravan, ~45 sec)*
> "Hi everyone! Thanks for being here. We're Group 16, and today we're excited to walk you through our completed project and live software demo for **Project 16: Context-Conditioned Causal Alignment for Scientific Literature**, or **CCST** for short.
>
> Our team includes Sri Harshith, Subhash, Tarun, Vedang, and myself, Shravan.
>
> Over the next 12 minutes, we'll keep things simple and show you what problem we noticed in scientific search, how our system fixes it under the hood, our test results across more than 2,000 real research papers, and then we'll switch over to a live demo of the working app."

---

### Slide 2 — Problem Definition *(Shravan, ~1 min 30 sec)*
> "So let's start with the problem. Suppose a scientist or student is searching across research papers using standard AI tools—like keyword search, Sentence-BERT, or even prompting an LLM. Right now, those tools fall into two super common traps.
>
> **First**, they miss **hidden analogies** across different fields. For example, a computer security paper might talk about malicious servers crossing a failure threshold and crashing a network, while a medical paper talks about unvaccinated rates crossing herd immunity and causing an outbreak. Those are the exact same cause-and-effect pattern! But because they use completely different jargon, existing search models miss them 95 to 100 percent of the time.
>
> **Second**, and honestly even more frustrating, standard AI creates **false contradiction alarms**—basically comparing apples to oranges. One study says a drug failed to lower glucose; another says the exact same drug lowered glucose. An AI tool reads those two sentences and immediately yells *Contradiction!* without realizing Paper A tested **mouse cells in a petri dish**, while Paper B tested **human patients in a clinic**. Across our tests, popular baselines made this mistake 38 to 44 percent of the time.
>
> So our goal was straightforward: instead of matching raw text similarity, extract the actual *Cause-to-Effect* chain **and** check the experiment conditions—who was tested, what dose, and in what setting—so scientists get real answers, not false alarms.
>
> I'll hand it over to Sri Harshith to show how our system is put together."

---

### Slide 3 — System Architecture: 4 Simple Steps *(Sri Harshith, ~1 min)*
> "Thanks Shravan. We designed our software around four clean steps so it stays modular and fast.
>
> **Step 1** reads raw paper abstracts and extracts structured Cause-to-Effect flows alongside experiment tags like species, dosage, and lab vs. human setting. We pulled over 16,600 mechanisms this way.
>
> **Step 2** is fast pre-filtering. Checking every possible pair out of thousands of papers would be super slow, so we combine keywords, meaning vectors, and role filters to cut out 96 percent of unrelated pairs right away—giving us a 25x speedup.
>
> **Step 3** runs our alignment engine, CCST, which checks both graph shape and context conditions in under 6 milliseconds per pair on ASU's SOL supercomputer GPU.
>
> **Step 4** outputs a clear verdict—whether two papers agree, contradict, have an analogy, or simply ran different experiments—along with exact quote highlights as proof."

---

### Slide 4 — Core Algorithms Explained Simply *(Sri Harshith, ~1 min 15 sec)*
> "Now without drowning anyone in equations, there are really just two smart ideas making CCST work.
>
> On the left is **Dual-View Graph Matching**. Scientific papers don't all have the same length or detail, so our algorithm uses partial matching—pairing shared steps cleanly without forcing extra steps to match nonsense. And we look at graphs in two ways: an *unsigned structure view* that ignores domain vocabulary so we can catch cross-domain analogies 100% of the time, and a *signed view* that checks whether an effect goes up or down so we catch true conflicts.
>
> On the right is our **Material Context Gate**. Before our system is ever allowed to label two papers a contradiction, it checks four conditions: Species, Setting (like petri dish vs. human trial), Dose, and Time. If those don't match—or if one paper forgot to mention them—the Context Gate automatically blocks the contradiction flag and classifies it as different experiment conditions instead.
>
> Plus, whenever there *is* a real conflict, it hands the user a plain-English Disagreement Certificate showing exactly which arrow flipped and which sentence it came from. Subhash will walk through our data and results next."

---

### Slide 5 — Datasets & Preprocessing *(Subhash, ~1 min)*
> "Thanks Sri Harshith. When we tested this, we made sure not to rely on toy examples—everything here runs on **real published papers**.
>
> On the left, we pulled **2,054 real papers** straight from public APIs: half—1,027 biomedical studies—from NCBI PubMed, and the other 1,027 papers from arXiv across security, computer networks, and control systems. For cleaning, we kept English abstracts with at least 50 words, screened out any retracted studies, and removed duplicates using MD5 hashes before extracting over 35,000 causal links.
>
> On the right, we built a balanced **240-pair evaluation benchmark** split evenly—60 pairs each—across Agreement, Cross-Domain Analogies, Context Differences, and True Contradictions. Three independent reviewers checked the labels and reached a Krippendorff's alpha agreement of 0.767, so we know our ground truth is solid."

---

### Slide 6 — Evaluation Results vs. Existing AI *(Subhash, ~1 min 30 sec)*
> "So how well did CCST actually perform compared to existing AI baselines? We ran a strict 60/20/20 train/dev/test split with 1,000 bootstrap runs on the SOL A100 GPU.
>
> Look at the left chart first—**False Contradictions**, where lower is better. Tools like Contriever, Sentence-BERT, and even a direct LLM judge falsely flagged different experiments as contradictions **40 to 44 percent of the time**. Our system brought that all the way down to **0.0 percent**—zero false alarms.
>
> Now look at the right chart—**Cross-Domain Analogies**, where higher is better. Standard keyword search and Sentence-BERT got **0 percent** because words across different fields don't overlap, and the LLM judge only caught 30 percent. CCST found **100 percent** of them.
>
> In the summary table below, our overall Macro F1 jumped to **43.9 percent**—nearly double the best baseline—while running in just **5.77 milliseconds per pair**, more than twice as fast as calling an LLM judge.
>
> Next, Tarun will show what happened when we tested turning individual parts of our system off."

---

### Slide 7 — Ablations: Why Every Component Matters *(Tarun, ~1 min)*
> "Thanks Subhash. To verify that our design wasn't just lucky—and that each part actually does real work—we ran component ablation tests.
>
> First, if we turn off our **Context Gate**, false contradiction alarms immediately spike from 0 percent up to **100 percent** on context-shifted studies, and overall F1 drops sharply.
>
> Second, if we remove the **Structure View** and only rely on signed wording, cross-domain analogy discovery crashes from 100 percent straight down to **0 percent**.
>
> And third, **100 percent** of our 240 output certificates map cleanly back to exact character-level quotes from the real abstracts, with statistically significant improvements at p < 0.001 over every baseline.
>
> Now Vedang is going to walk us through the live interactive UI demo."

---

### Slide 8 — Live UI Demo (Part 1: Intuitive Claim Comparison) *(Vedang, ~1 min 30 sec)*
> *(Switch over to `demo/ccst_interactive_demo.html` open in your browser)*
>
> "Thanks Tarun! Let's jump right into the live app so you can see how simple it is to explore.
>
> We wanted this UI to be super clean and readable in three steps.
>
> **Step 1** at the top lets you pick a real test case—or click `+ Try Custom Claims` at the far right to type in your own sentences. Let's look at the first one right now: **Metformin & Blood Sugar**.
>
> Notice the two clean cards in the middle. On the left, **Paper A** says Metformin failed to suppress glucose output—and right below it, our system highlights why: *Mice*, *Petri Dish (In Vitro)*, *10 mM high dose*.
>
> On the right, **Paper B** says Metformin significantly lowered blood glucose—highlighted with *Humans*, *Clinical Trial (In Vivo)*, *500 mg daily*.
>
> Look down at **Step 3: Final Verdict** in amber: **NOT A CONTRADICTION — Different Experiment Conditions**, explaining in one simple sentence why the Context Gate stopped a false alarm."

---

### Slide 9 — Live UI Demo (Part 2: Standard AI Toggle & Proof) *(Vedang, ~1 min 15 sec)*
> *(Stay in `ccst_interactive_demo.html` to click live!)*
>
> "Now here's my favorite part to demonstrate live. Right under Step 1, see this toggle between **Our System** and **Standard AI**? Watch what happens when I click **`Standard AI (Ignores Context)`**...
> *(Click **`✗ Standard AI (Ignores Context)`**)*
>
> Boom—the whole verdict box turns bright red: **`FALSE CONTRADICTION ALERT (91.5% confident)`**! Because normal AI ignores those yellow mouse-vs-human tags, it misleads users into thinking these two studies clash. When we switch **`Our System`** back on, that false alarm disappears.
>
> And let's quickly click the third tab, **Security vs. Viruses**. Even though Paper A talks about malicious computer servers and Paper B talks about unvaccinated populations, CCST spots the identical *Threshold → Spread Cascade* shape and flags a **Cross-Domain Analogy at 92.4% confidence**.
>
> Finally, if a reviewer wants technical proof, one click on **`+ Show Technical Proof & Disagreement Certificate`** expands the calibrated 5-way confidence bars and auditable JSON certificate without cluttering the screen.
>
> I'll pass it back to Tarun briefly for our project timeline and team breakdown."

---

### Slide 10 — Project Plan & Division of Work *(Tarun, ~45 sec)*
> "Thanks Vedang. Wrapping up with how we planned and executed the semester:
>
> In September we scoped the problem and built the live PubMed and arXiv pipelines; in October we built the causal extractor, human benchmark, and baselines; in early November we finished the alignment solver, classifier, and SOL A100 evaluations; and this past week we packaged the simplified demo UI and repository cleanly—hitting every deadline on schedule.
>
> Everyone on Team 16 owned a clear piece of the stack: Shravan led the dual-view alignment theory; Sri Harshith built the data pipelines and reliability audit; Subhash handled extraction and baselines; I worked on solver optimization and 25x indexing; and Vedang built the classifier, certificate builder, demo UI, and SOL GPU deployment."

---

### Slide 11 — Thank You & GitHub Repository *(Vedang, ~30 sec)*
> "And to close out—all of our software is cleanly organized on GitHub inside `/code`, `/data`, `/evaluations`, and `/demo`, with 32 out of 32 automated tests passing and a one-line CLI runner (`python3 code/run_demo.py`).
>
> Our GitHub repository link is right here on the final slide. Thank you so much for your attention—we'd love to take any questions!"

---

## Quick Q&A Cheat-Sheet (Short & Natural Answers)

- **Why not just ask ChatGPT / an LLM to compare two papers?**
  > "LLMs read surface text without strictly checking missing context—on our benchmark, a direct LLM judge still triggered 40% false contradictions, only caught 30% of analogies, and ran over twice as slow as CCST. Plus CCST links every single decision directly back to exact character quotes rather than generating black-box text."

- **What happens if a paper doesn't mention its species or dosage at all?**
  > "Great question—instead of assuming two papers match when info is missing, our system explicitly marks missing fields as *Unknown*. Our Context Gate blocks a contradiction label unless compatible conditions are actually verified, or cleanly abstains under *Insufficient Comparability*."

- **Can we run this live on any two papers, not just the 4 presets?**
  > "Yes! In our live web UI you can click **`+ Try Custom Claims`** to test custom claims on the spot, and in Python you can run `python3 code/run_demo.py` or invoke the Phase 3–5 extractor + solver on any abstract from PubMed or arXiv."

- **Why is Macro F1 43.9% even though False Contradictions are 0% and Analogy Recall is 100%?**
  > "Because our 5-way task includes a strict **Abstention / Insufficient Comparability** class and conservative context thresholds—our model intentionally refuses to over-claim contradictions unless context matches strictly, trading aggressive guessing for zero false alarms (more than double Sentence-BERT's 24.5% Macro F1)."
