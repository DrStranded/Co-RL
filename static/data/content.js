// Co-RL project page — content data.
// NOTE: index.html hard-codes the hero authors/affiliations and the BibTeX block so the page
// works without JavaScript; the fields here are the canonical machine-readable copies. If one
// changes, change both.
// Every number quoted here must exist in the paper source. See SPEC.md content rules.
window.siteContent = {
  title: "Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL",
  shortTitle: "Co-RL",
  arxivId: "2608.17253",
  arxivUrl: "https://arxiv.org/abs/2608.17253",
  pdfUrl: "https://arxiv.org/pdf/2608.17253",
  codeUrl: "https://github.com/DrStranded/Co-RL",

  authors: [
    { name: "Yunhao Yang", mark: "1,*" },
    { name: "Yuexin Bian", mark: "2,*" },
    { name: "Yunjie Tian", mark: "4" },
    { name: "Di Fu", mark: "4" },
    { name: "Tianjin Huang", mark: "3" },
    { name: "Yuanyuan Shi", mark: "2" },
    { name: "Ziang Xiao", mark: "1" },
    { name: "Nuno Vasconcelos", mark: "2" },
    { name: "Yijiang Li", mark: "2,†" }
  ],
  affiliations: [
    "1 Johns Hopkins University",
    "2 UC San Diego",
    "3 University of Exeter",
    "4 Independent Researcher"
  ],
  authorNotes: "* equal contribution · † project lead",

  // Rung rail (#ladder). kappa/c ranges from tab:dec_pool; averages from tab:llm3b/78b.
  rungs: [
    {
      id: "seed",
      index: "0",
      label: "Seed only",
      kappa: "κ 0.51–0.58",
      badge: "Diagnostic only",
      badgeTone: "muted",
      what: "Same weights, different sampling seed. The two views share every parameter and differ in generation noise alone.",
      measured: "κ 0.51–0.58, complementarity 19.0–24.6% across four seed-only pairs.",
      trained: "Never trained. This rung exists as the capability control that makes the anchor comparison work: the seed-only row is a model paired with itself, so capability is identical and the partner is the only variable.",
      result: ""
    },
    {
      id: "same-family",
      index: "1",
      label: "Same family",
      kappa: "κ 0.51–0.52",
      badge: "Co-RL (Same family)",
      badgeTone: "accent",
      what: "Two agents initialized from one lineage, across sizes or generations. Decoupled parameters and optimizer states, but a shared model stack.",
      measured: "κ 0.51–0.52, complementarity 24.2–24.4% — indistinguishable from seed-only pairs.",
      trained: "The readily accessible setting: no second model family needed.",
      result: "Already lifts the seven-benchmark average by 8.0% on Qwen2.5-3B and 4.0% on Llama-3.2-3B-Instruct over the base models."
    },
    {
      id: "diff-family",
      index: "2",
      label: "Different family",
      kappa: "κ 0.31–0.42",
      badge: "Co-RL (Different family)",
      badgeTone: "accent",
      what: "Separate architectures, tokenizers, vocabularies, and pretraining corpora — e.g. Qwen2.5 × Llama 3. Every design choice made independently.",
      measured: "κ 0.31–0.42, complementarity 29.4–32.8%. Every different-family pair sits below every same-family pair, with no overlap.",
      trained: "The main setting of the paper.",
      result: "Beats the strongest self-rewarding baseline on all four text backbones."
    },
    {
      id: "diff-data",
      index: "3",
      label: "Different family +",
      kappa: "+ decoupled data",
      badge: "Co-RL (Different family+)",
      badgeTone: "accent",
      what: "One agent trains on the original MATH prompt, the other on a DeepSeek-V3 rewrite that preserves the answer and the sample order while recasting the problem into a different concrete scenario.",
      measured: "In a sample of 300 rewrite pairs, every rewrite preserved the answer and the row alignment.",
      trained: "The strongest label-free variant.",
      result: "Best label-free average on all four text backbones. Caveat, stated plainly: the rewrite touches training prompts only and never supplies an answer — and rung 2, with no rewriting at all, already beats every self-rewarding baseline.",
      rephrase: {
        original: "How many vertical asymptotes does the graph of y = 2/(x\u00b2+x\u22126) have?",
        rephrased: "The function f(t) = 2/(t\u00b2+t\u22126) describes the temperature of a chemical reaction over time t. How many vertical asymptotes appear on the graph of this function?"
      }
    },
    {
      id: "modality",
      index: "4",
      label: "Different modality",
      kappa: "vision-language",
      badge: "Not measured by the probe",
      badgeTone: "muted",
      what: "Vision-language families differ in the vision encoder as well as the language backbone: Qwen2.5-VL pairs a natively trained dynamic-resolution ViT, InternVL adopts InternViT, Gemma 3 uses SigLIP.",
      measured: "The κ probe covers text base models only, so this rung is argued from architecture and confirmed downstream.",
      trained: "Trained with five vision-language models from 2B to 12B: the 2B\u20133B pair on both open-r1 and MMR1, the 7B\u201312B trio on open-r1.",
      result: "Gains of 2.3–7.2% on the four-benchmark multimodal suite. No Different family+ variant exists for vision-language models — data decoupling has only been run on text."
    }
  ],

  bibtex: "@article{yang2026corl,\n  title   = {Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL},\n  author  = {Yang, Yunhao and Bian, Yuexin and Tian, Yunjie and Fu, Di and Huang, Tianjin and Shi, Yuanyuan and Xiao, Ziang and Vasconcelos, Nuno and Li, Yijiang},\n  journal = {arXiv preprint arXiv:2608.17253},\n  year    = {2026},\n  eprint  = {2608.17253},\n  archivePrefix = {arXiv},\n  primaryClass  = {cs.LG}\n}"
};
