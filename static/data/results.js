// Co-RL project page — result data.
// GENERATED from the paper's LaTeX tables (neurips_2026.tex + appendix.tex) by extract_tables.py
// + normalize.py. Do not edit numbers by hand; regenerate from the .tex source.
// Values are STRINGS carrying the paper's exact formatting; parseFloat() before arithmetic.
// marks: "b" = bold (best label-free), "u" = underline (second best), "" = none.
// HARD PARTITION: vlmSmall blocks are keyed by (backbone, dataset). MMR1 uses the corrected
// multiple-choice grader and open-r1 the legacy grader, so the two datasets are NEVER
// compared with each other, and the UI must never place them side by side.
// The CoMAS table uses CoMAS's suite/driver/graders; ensemble tables use maj@8 at T=0.6.
// Neither shares an axis with the seven-benchmark tables.
window.siteResults = {
  "benchText": [
    "GSM8K",
    "MATH500",
    "AMC",
    "HumanEval",
    "GPQA",
    "MBPP",
    "LiveCodeBench",
    "Avg"
  ],
  "benchVLM": [
    "MathVision",
    "MathVerse",
    "MathVista",
    "We-Math",
    "Avg"
  ],
  "llmMain": [
    {
      "backbone": "Qwen2.5-3B",
      "tier": "3B",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "73.4",
            "56.6",
            "28.9",
            "39.0",
            "21.2",
            "52.2",
            "13.7",
            "40.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "76.2",
            "64.6",
            "36.1",
            "65.2",
            "20.7",
            "54.4",
            "14.5",
            "47.4"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "80.4",
            "66.4",
            "31.3",
            "63.4",
            "22.2",
            "51.8",
            "15.9",
            "47.3"
          ],
          "marks": [
            "u",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "RENT",
          "vals": [
            "75.6",
            "62.8",
            "31.3",
            "59.2",
            "18.2",
            "52.4",
            "14.5",
            "44.9"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Intuitor",
          "vals": [
            "74.9",
            "64.2",
            "26.5",
            "59.8",
            "27.3",
            "50.4",
            "16.4",
            "45.6"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "b",
            "",
            "u",
            ""
          ]
        },
        {
          "method": "Co-rewarding-II",
          "vals": [
            "75.5",
            "63.4",
            "30.1",
            "61.0",
            "24.8",
            "53.2",
            "11.0",
            "45.6"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Co-RL (Same family)",
          "vals": [
            "78.5",
            "66.0",
            "37.4",
            "65.8",
            "22.2",
            "56.0",
            "15.2",
            "48.7"
          ],
          "marks": [
            "",
            "",
            "b",
            "b",
            "",
            "u",
            "",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "80.1",
            "66.8",
            "33.7",
            "64.0",
            "22.7",
            "56.8",
            "15.2",
            "48.5"
          ],
          "marks": [
            "",
            "b",
            "",
            "u",
            "",
            "b",
            "",
            ""
          ]
        },
        {
          "method": "Co-RL (Different family+)",
          "vals": [
            "81.0",
            "66.6",
            "36.1",
            "62.8",
            "25.8",
            "55.6",
            "17.2",
            "49.3"
          ],
          "marks": [
            "b",
            "u",
            "u",
            "",
            "u",
            "",
            "b",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "Llama-3.2-3B-Instruct",
      "tier": "3B",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "73.6",
            "43.8",
            "18.1",
            "51.2",
            "21.2",
            "50.8",
            "12.0",
            "38.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "78.8",
            "53.8",
            "25.3",
            "60.4",
            "20.7",
            "50.2",
            "12.1",
            "43.0"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "77.9",
            "50.2",
            "26.5",
            "59.2",
            "24.8",
            "51.2",
            "12.0",
            "43.1"
          ],
          "marks": [
            "",
            "",
            "",
            "b",
            "b",
            "u",
            "",
            ""
          ]
        },
        {
          "method": "RENT",
          "vals": [
            "75.4",
            "45.2",
            "12.0",
            "59.2",
            "17.7",
            "49.4",
            "11.5",
            "38.6"
          ],
          "marks": [
            "",
            "",
            "",
            "b",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Intuitor",
          "vals": [
            "75.8",
            "40.8",
            "21.7",
            "54.3",
            "21.7",
            "51.4",
            "12.0",
            "39.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "b",
            "",
            ""
          ]
        },
        {
          "method": "Co-rewarding-II",
          "vals": [
            "75.4",
            "53.4",
            "24.1",
            "54.9",
            "23.7",
            "49.2",
            "12.1",
            "41.8"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "u",
            "",
            "u",
            ""
          ]
        },
        {
          "method": "Co-RL (Same family)",
          "vals": [
            "78.4",
            "52.4",
            "26.5",
            "57.9",
            "21.7",
            "49.6",
            "12.4",
            "42.7"
          ],
          "marks": [
            "u",
            "",
            "",
            "u",
            "",
            "",
            "b",
            ""
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "80.5",
            "56.2",
            "27.7",
            "59.2",
            "21.2",
            "50.4",
            "11.0",
            "43.7"
          ],
          "marks": [
            "b",
            "b",
            "u",
            "b",
            "",
            "",
            "",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family+)",
          "vals": [
            "78.4",
            "55.2",
            "30.1",
            "59.2",
            "22.2",
            "50.4",
            "12.0",
            "43.9"
          ],
          "marks": [
            "u",
            "u",
            "b",
            "b",
            "",
            "",
            "",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "Qwen2.5-7B",
      "tier": "7B/8B",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "82.9",
            "70.0",
            "39.8",
            "47.6",
            "18.7",
            "62.8",
            "21.1",
            "49.0"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "84.8",
            "77.6",
            "49.4",
            "56.1",
            "23.7",
            "64.4",
            "25.5",
            "54.5"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "80.6",
            "74.8",
            "39.8",
            "51.8",
            "25.8",
            "65.4",
            "23.9",
            "51.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "u",
            "",
            ""
          ]
        },
        {
          "method": "RENT",
          "vals": [
            "78.8",
            "75.4",
            "47.0",
            "50.6",
            "29.8",
            "61.6",
            "26.2",
            "52.8"
          ],
          "marks": [
            "",
            "b",
            "b",
            "",
            "u",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Intuitor",
          "vals": [
            "82.9",
            "75.4",
            "41.0",
            "51.8",
            "28.3",
            "64.0",
            "24.8",
            "52.6"
          ],
          "marks": [
            "b",
            "b",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Co-rewarding-II",
          "vals": [
            "81.9",
            "72.6",
            "43.4",
            "52.4",
            "26.8",
            "64.0",
            "25.9",
            "52.4"
          ],
          "marks": [
            "u",
            "",
            "",
            "u",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Co-RL (Same family)",
          "vals": [
            "78.9",
            "74.6",
            "41.0",
            "52.4",
            "25.8",
            "61.8",
            "25.0",
            "51.4"
          ],
          "marks": [
            "",
            "",
            "",
            "u",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "81.3",
            "75.2",
            "44.6",
            "52.4",
            "26.3",
            "65.6",
            "26.5",
            "53.1"
          ],
          "marks": [
            "",
            "u",
            "u",
            "u",
            "",
            "b",
            "u",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family+)",
          "vals": [
            "80.2",
            "74.4",
            "38.6",
            "54.3",
            "37.9",
            "63.2",
            "26.6",
            "53.6"
          ],
          "marks": [
            "",
            "",
            "",
            "b",
            "b",
            "",
            "b",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "Llama-3.1-8B-Instruct",
      "tier": "7B/8B",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "82.9",
            "49.6",
            "18.1",
            "65.2",
            "22.2",
            "58.4",
            "16.8",
            "44.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "82.7",
            "53.2",
            "25.3",
            "64.0",
            "30.3",
            "59.2",
            "15.2",
            "47.1"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "83.9",
            "51.0",
            "27.7",
            "64.6",
            "21.2",
            "58.2",
            "16.3",
            "46.1"
          ],
          "marks": [
            "",
            "",
            "b",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "RENT",
          "vals": [
            "79.5",
            "48.2",
            "21.7",
            "67.7",
            "19.7",
            "60.0",
            "16.0",
            "44.7"
          ],
          "marks": [
            "",
            "",
            "",
            "u",
            "",
            "u",
            "",
            ""
          ]
        },
        {
          "method": "Intuitor",
          "vals": [
            "79.7",
            "45.8",
            "21.7",
            "65.8",
            "26.8",
            "58.0",
            "16.1",
            "44.8"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "u",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Co-rewarding-II",
          "vals": [
            "84.7",
            "52.0",
            "24.1",
            "67.1",
            "22.2",
            "59.8",
            "16.5",
            "46.6"
          ],
          "marks": [
            "u",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Co-RL (Same family)",
          "vals": [
            "85.4",
            "51.4",
            "22.9",
            "68.3",
            "23.2",
            "60.4",
            "15.8",
            "46.8"
          ],
          "marks": [
            "b",
            "",
            "",
            "b",
            "",
            "b",
            "",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "83.6",
            "54.8",
            "27.7",
            "67.7",
            "18.2",
            "57.6",
            "17.7",
            "46.8"
          ],
          "marks": [
            "",
            "u",
            "b",
            "u",
            "",
            "",
            "b",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family+)",
          "vals": [
            "85.4",
            "55.6",
            "26.5",
            "64.6",
            "27.3",
            "57.2",
            "17.1",
            "47.7"
          ],
          "marks": [
            "b",
            "b",
            "u",
            "",
            "b",
            "",
            "u",
            "b"
          ]
        }
      ]
    }
  ],
  "n3": [
    {
      "model": "Qwen2.5-3B",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "73.4",
            "56.6",
            "28.9",
            "39.0",
            "21.2",
            "52.2",
            "13.7",
            "40.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "76.2",
            "64.6",
            "36.1",
            "65.2",
            "20.7",
            "54.4",
            "14.5",
            "47.4"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "80.4",
            "66.4",
            "31.3",
            "63.4",
            "22.2",
            "51.8",
            "15.9",
            "47.3"
          ],
          "marks": [
            "b",
            "b",
            "u",
            "u",
            "u",
            "u",
            "b",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "79.8",
            "66.3",
            "33.6",
            "64.6",
            "23.2",
            "56.0",
            "15.8",
            "48.5"
          ],
          "marks": [
            "u",
            "u",
            "b",
            "b",
            "b",
            "b",
            "u",
            "b"
          ]
        }
      ]
    },
    {
      "model": "Llama-3.2-3B-Instruct",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "73.6",
            "43.8",
            "18.1",
            "51.2",
            "21.2",
            "50.8",
            "12.0",
            "38.7"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "78.8",
            "53.8",
            "25.3",
            "60.4",
            "20.7",
            "50.2",
            "12.1",
            "43.0"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "77.9",
            "50.2",
            "26.5",
            "59.2",
            "24.8",
            "51.2",
            "12.0",
            "43.1"
          ],
          "marks": [
            "b",
            "u",
            "u",
            "u",
            "u",
            "b",
            "b",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "77.8",
            "54.2",
            "28.8",
            "64.4",
            "25.1",
            "50.9",
            "11.7",
            "44.7"
          ],
          "marks": [
            "u",
            "b",
            "b",
            "b",
            "b",
            "u",
            "u",
            "b"
          ]
        }
      ]
    },
    {
      "model": "Qwen3-1.7B",
      "rows": [
        {
          "method": "Base",
          "vals": [
            "67.0",
            "60.9",
            "27.5",
            "40.0",
            "15.3",
            "50.6",
            "12.4",
            "39.1"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "GT-Reward",
          "vals": [
            "67.1",
            "67.0",
            "34.3",
            "70.1",
            "25.2",
            "51.2",
            "15.2",
            "47.2"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "70.3",
            "67.6",
            "32.1",
            "69.5",
            "24.8",
            "52.0",
            "15.1",
            "47.3"
          ],
          "marks": [
            "b",
            "b",
            "u",
            "b",
            "u",
            "u",
            "u",
            "b"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "69.3",
            "67.6",
            "32.7",
            "64.2",
            "27.1",
            "54.6",
            "15.3",
            "47.3"
          ],
          "marks": [
            "u",
            "b",
            "b",
            "u",
            "b",
            "b",
            "b",
            "b"
          ]
        }
      ]
    }
  ],
  "vlmSmall": [
    {
      "backbone": "InternVL-3.5-2B",
      "dataset": "open-r1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "26.55",
            "35.33",
            "59.60",
            "59.31",
            "45.20"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "24.77",
            "34.21",
            "55.60",
            "57.87",
            "43.11"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "25.86",
            "34.24",
            "57.60",
            "62.47",
            "45.04"
          ],
          "marks": [
            "u",
            "u",
            "u",
            "b",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "26.25",
            "34.92",
            "58.90",
            "61.55",
            "45.40"
          ],
          "marks": [
            "b",
            "b",
            "b",
            "u",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "InternVL-3.5-2B",
      "dataset": "MMR1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "25.99",
            "34.37",
            "59.00",
            "59.25",
            "44.65"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "24.77",
            "34.21",
            "55.60",
            "57.87",
            "43.11"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "26.38",
            "35.36",
            "57.70",
            "61.78",
            "45.30"
          ],
          "marks": [
            "b",
            "b",
            "u",
            "b",
            "b"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "26.05",
            "34.80",
            "58.60",
            "61.15",
            "45.15"
          ],
          "marks": [
            "u",
            "u",
            "b",
            "u",
            "u"
          ]
        }
      ]
    },
    {
      "backbone": "Qwen2.5-VL-3B",
      "dataset": "open-r1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "21.71",
            "31.29",
            "60.90",
            "57.99",
            "42.97"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "18.55",
            "26.04",
            "52.70",
            "51.67",
            "37.24"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "21.15",
            "30.05",
            "57.40",
            "61.55",
            "42.54"
          ],
          "marks": [
            "u",
            "u",
            "u",
            "u",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "21.94",
            "30.48",
            "60.20",
            "62.93",
            "43.89"
          ],
          "marks": [
            "b",
            "b",
            "b",
            "b",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "Qwen2.5-VL-3B",
      "dataset": "MMR1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "19.57",
            "27.34",
            "59.40",
            "57.82",
            "41.03"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "18.55",
            "26.04",
            "52.70",
            "51.67",
            "37.24"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "17.99",
            "24.72",
            "56.30",
            "52.87",
            "37.97"
          ],
          "marks": [
            "u",
            "u",
            "u",
            "u",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "21.05",
            "28.91",
            "57.20",
            "57.30",
            "41.12"
          ],
          "marks": [
            "b",
            "b",
            "b",
            "b",
            "b"
          ]
        }
      ]
    }
  ],
  "vlmLarge": [
    {
      "backbone": "Qwen2.5-VL-7B",
      "dataset": "open-r1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "26.74",
            "41.07",
            "71.90",
            "67.01",
            "51.68"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "23.36",
            "33.32",
            "56.60",
            "62.47",
            "43.94"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "23.62",
            "37.26",
            "69.40",
            "65.23",
            "48.88"
          ],
          "marks": [
            "u",
            "u",
            "u",
            "u",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "26.87",
            "38.43",
            "71.00",
            "68.22",
            "51.13"
          ],
          "marks": [
            "b",
            "b",
            "b",
            "b",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "InternVL-3.5-8B",
      "dataset": "open-r1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "37.24",
            "43.35",
            "69.30",
            "73.51",
            "55.85"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "29.21",
            "36.65",
            "65.70",
            "60.69",
            "48.06"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "35.07",
            "41.24",
            "68.60",
            "71.72",
            "54.16"
          ],
          "marks": [
            "u",
            "b",
            "u",
            "b",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "35.30",
            "40.74",
            "70.60",
            "70.98",
            "54.40"
          ],
          "marks": [
            "b",
            "u",
            "b",
            "u",
            "b"
          ]
        }
      ]
    },
    {
      "backbone": "Gemma-3-12B",
      "dataset": "open-r1",
      "rows": [
        {
          "method": "GT-Reward",
          "vals": [
            "30.89",
            "33.63",
            "56.90",
            "59.25",
            "45.17"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "Base",
          "vals": [
            "27.20",
            "32.70",
            "46.70",
            "60.50",
            "41.78"
          ],
          "marks": [
            "",
            "",
            "",
            "",
            ""
          ]
        },
        {
          "method": "TTRL",
          "vals": [
            "27.93",
            "36.37",
            "54.70",
            "58.79",
            "44.45"
          ],
          "marks": [
            "u",
            "b",
            "u",
            "u",
            "u"
          ]
        },
        {
          "method": "Co-RL (Different family)",
          "vals": [
            "32.01",
            "35.91",
            "55.60",
            "66.72",
            "47.56"
          ],
          "marks": [
            "b",
            "u",
            "b",
            "b",
            "b"
          ]
        }
      ]
    }
  ],
  "comas": {
    "header": [
      "GSM8K",
      "MATH-500",
      "HumanEval",
      "MBPP",
      "MMLU",
      "GPQA",
      "SciBench",
      "Avg"
    ],
    "rows": [
      {
        "method": "Base",
        "vals": [
          "85.40",
          "55.00",
          "73.78",
          "55.80",
          "63.20",
          "28.79",
          "36.47",
          "56.92"
        ],
        "marks": [
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          ""
        ]
      },
      {
        "method": "MAPoRL",
        "vals": [
          "85.80",
          "55.40",
          "75.61",
          "57.00",
          "63.20",
          "31.47",
          "39.08",
          "58.22"
        ],
        "marks": [
          "",
          "",
          "",
          "",
          "",
          "b",
          "b",
          ""
        ]
      },
      {
        "method": "TTRL",
        "vals": [
          "88.20",
          "56.80",
          "73.78",
          "59.00",
          "63.80",
          "27.23",
          "38.48",
          "58.18"
        ],
        "marks": [
          "u",
          "u",
          "",
          "",
          "",
          "",
          "u",
          ""
        ]
      },
      {
        "method": "CoMAS",
        "vals": [
          "87.20",
          "55.80",
          "77.44",
          "59.20",
          "65.60",
          "29.69",
          "37.68",
          "58.94"
        ],
        "marks": [
          "",
          "",
          "u",
          "u",
          "u",
          "u",
          "",
          "u"
        ]
      },
      {
        "method": "Co-RL (Different family)",
        "vals": [
          "89.5",
          "68.6",
          "82.32",
          "68.00",
          "65.80",
          "29.69",
          "36.87",
          "62.97"
        ],
        "marks": [
          "b",
          "b",
          "b",
          "b",
          "b",
          "u",
          "",
          "b"
        ]
      }
    ]
  },
  "ensLLM": [
    {
      "group": null,
      "setting": "TTRL (Qwen2.5-3B)",
      "vals": [
        "88.2",
        "68.8",
        "39.8",
        "65.6"
      ],
      "marks": [
        "u",
        "",
        "b",
        ""
      ]
    },
    {
      "group": null,
      "setting": "TTRL (Llama-3.2-3B)",
      "vals": [
        "65.7",
        "56.0",
        "27.7",
        "49.8"
      ],
      "marks": [
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": null,
      "setting": "TTRL (ensemble)",
      "vals": [
        "88.2",
        "68.0",
        "38.6",
        "64.9"
      ],
      "marks": [
        "u",
        "",
        "u",
        ""
      ]
    },
    {
      "group": null,
      "setting": "Co-RL (Qwen2.5-3B)",
      "vals": [
        "87.4",
        "72.8",
        "37.4",
        "65.9"
      ],
      "marks": [
        "",
        "b",
        "",
        "u"
      ]
    },
    {
      "group": null,
      "setting": "Co-RL (Llama-3.2-3B)",
      "vals": [
        "87.3",
        "58.8",
        "33.7",
        "59.9"
      ],
      "marks": [
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": null,
      "setting": "Co-RL (ensemble)",
      "vals": [
        "90.1",
        "70.8",
        "39.8",
        "66.9"
      ],
      "marks": [
        "b",
        "u",
        "b",
        "b"
      ]
    }
  ],
  "ensVLM": [
    {
      "group": "open-r1",
      "setting": "TTRL (Qwen2.5-VL)",
      "vals": [
        "22.96",
        "31.45",
        "61.10",
        "63.39",
        "44.73"
      ],
      "marks": [
        "",
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": "open-r1",
      "setting": "TTRL (InternVL3.5)",
      "vals": [
        "29.67",
        "38.91",
        "62.30",
        "67.24",
        "49.53"
      ],
      "marks": [
        "u",
        "b",
        "",
        "",
        ""
      ]
    },
    {
      "group": "open-r1",
      "setting": "TTRL (ensemble)",
      "vals": [
        "27.24",
        "35.13",
        "65.40",
        "67.41",
        "48.80"
      ],
      "marks": [
        "",
        "",
        "u",
        "",
        ""
      ]
    },
    {
      "group": "open-r1",
      "setting": "Co-RL (Qwen2.5-VL)",
      "vals": [
        "25.43",
        "35.66",
        "64.80",
        "65.80",
        "47.92"
      ],
      "marks": [
        "",
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": "open-r1",
      "setting": "Co-RL (InternVL3.5)",
      "vals": [
        "30.46",
        "38.60",
        "63.30",
        "67.87",
        "50.06"
      ],
      "marks": [
        "b",
        "u",
        "",
        "u",
        "u"
      ]
    },
    {
      "group": "open-r1",
      "setting": "Co-RL (ensemble)",
      "vals": [
        "28.95",
        "38.48",
        "67.00",
        "69.08",
        "50.88"
      ],
      "marks": [
        "",
        "",
        "b",
        "b",
        "b"
      ]
    },
    {
      "group": "MMR1",
      "setting": "TTRL (Qwen2.5-VL)",
      "vals": [
        "17.27",
        "30.71",
        "63.40",
        "60.57",
        "42.99"
      ],
      "marks": [
        "",
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": "MMR1",
      "setting": "TTRL (InternVL3.5)",
      "vals": [
        "28.78",
        "39.47",
        "63.70",
        "66.90",
        "49.71"
      ],
      "marks": [
        "",
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": "MMR1",
      "setting": "TTRL (ensemble)",
      "vals": [
        "25.53",
        "37.77",
        "67.00",
        "66.44",
        "49.19"
      ],
      "marks": [
        "",
        "",
        "u",
        "",
        ""
      ]
    },
    {
      "group": "MMR1",
      "setting": "Co-RL (Qwen2.5-VL)",
      "vals": [
        "25.86",
        "34.59",
        "66.00",
        "64.94",
        "47.85"
      ],
      "marks": [
        "",
        "",
        "",
        "",
        ""
      ]
    },
    {
      "group": "MMR1",
      "setting": "Co-RL (InternVL3.5)",
      "vals": [
        "30.79",
        "40.94",
        "65.30",
        "67.53",
        "51.14"
      ],
      "marks": [
        "b",
        "b",
        "",
        "u",
        "u"
      ]
    },
    {
      "group": "MMR1",
      "setting": "Co-RL (ensemble)",
      "vals": [
        "30.49",
        "39.75",
        "69.40",
        "69.54",
        "52.30"
      ],
      "marks": [
        "u",
        "u",
        "b",
        "b",
        "b"
      ]
    }
  ],
  "decPool": [
    {
      "tier": "3B tier",
      "level": "different family",
      "pair": "Llama-3.2-3B \u00d7 Phi-3.5-mini",
      "kappa": 0.31,
      "c": 32.8,
      "w": 3.0,
      "u": 53.0
    },
    {
      "tier": "3B tier",
      "level": "different family",
      "pair": "Qwen2.5-3B \u00d7 Llama-3.2-3B",
      "kappa": 0.38,
      "c": 31.2,
      "w": 2.4,
      "u": 63.0
    },
    {
      "tier": "3B tier",
      "level": "different family",
      "pair": "Qwen2.5-3B \u00d7 Phi-3.5-mini",
      "kappa": 0.38,
      "c": 31.2,
      "w": 4.0,
      "u": 55.4
    },
    {
      "tier": "3B tier",
      "level": "different family",
      "pair": "Qwen2.5-3B \u00d7 MiniCPM3-4B",
      "kappa": 0.41,
      "c": 29.4,
      "w": 4.4,
      "u": 60.4
    },
    {
      "tier": "3B tier",
      "level": "same family",
      "pair": "Qwen2.5-3B \u00d7 Qwen3-1.7B-Base",
      "kappa": 0.52,
      "c": 24.2,
      "w": 4.2,
      "u": 63.2
    },
    {
      "tier": "3B tier",
      "level": "seed only",
      "pair": "Qwen3-1.7B-Base \u00d7 itself",
      "kappa": 0.52,
      "c": 24.0,
      "w": 5.0,
      "u": 66.4
    },
    {
      "tier": "3B tier",
      "level": "seed only",
      "pair": "Qwen2.5-3B \u00d7 itself",
      "kappa": 0.56,
      "c": 22.0,
      "w": 4.4,
      "u": 62.6
    },
    {
      "tier": "7B tier",
      "level": "different family",
      "pair": "Qwen2.5-7B \u00d7 Llama-3.1-8B",
      "kappa": 0.42,
      "c": 29.4,
      "w": 1.8,
      "u": 71.4
    },
    {
      "tier": "7B tier",
      "level": "same family",
      "pair": "Qwen2.5-7B \u00d7 Qwen2.5-3B",
      "kappa": 0.51,
      "c": 24.2,
      "w": 3.8,
      "u": 69.8
    },
    {
      "tier": "7B tier",
      "level": "same family",
      "pair": "Qwen2.5-7B \u00d7 Qwen3-1.7B-Base",
      "kappa": 0.51,
      "c": 24.4,
      "w": 4.0,
      "u": 70.4
    },
    {
      "tier": "7B tier",
      "level": "seed only",
      "pair": "Llama-3.1-8B \u00d7 itself",
      "kappa": 0.51,
      "c": 24.6,
      "w": 3.0,
      "u": 62.0
    },
    {
      "tier": "7B tier",
      "level": "seed only",
      "pair": "Qwen2.5-7B \u00d7 itself",
      "kappa": 0.58,
      "c": 19.0,
      "w": 5.2,
      "u": 74.6
    }
  ],
  "decAnchor": [
    {
      "anchor": "Qwen2.5-3B",
      "partner": "itself, new seed",
      "level": "seed only",
      "kappa": 0.56,
      "c": 22.0,
      "w": 4.4
    },
    {
      "anchor": "Qwen2.5-3B",
      "partner": "Qwen3-1.7B-Base",
      "level": "same family",
      "kappa": 0.52,
      "c": 24.2,
      "w": 4.2
    },
    {
      "anchor": "Qwen2.5-3B",
      "partner": "MiniCPM3-4B",
      "level": "different family",
      "kappa": 0.41,
      "c": 29.4,
      "w": 4.4
    },
    {
      "anchor": "Qwen2.5-3B",
      "partner": "Phi-3.5-mini",
      "level": "different family",
      "kappa": 0.38,
      "c": 31.2,
      "w": 4.0
    },
    {
      "anchor": "Qwen2.5-3B",
      "partner": "Llama-3.2-3B",
      "level": "different family",
      "kappa": 0.38,
      "c": 31.2,
      "w": 2.4
    },
    {
      "anchor": "Qwen2.5-7B",
      "partner": "itself, new seed",
      "level": "seed only",
      "kappa": 0.58,
      "c": 19.0,
      "w": 5.2
    },
    {
      "anchor": "Qwen2.5-7B",
      "partner": "Qwen2.5-3B",
      "level": "same family",
      "kappa": 0.51,
      "c": 24.2,
      "w": 3.8
    },
    {
      "anchor": "Qwen2.5-7B",
      "partner": "Qwen3-1.7B-Base",
      "level": "same family",
      "kappa": 0.51,
      "c": 24.4,
      "w": 4.0
    },
    {
      "anchor": "Qwen2.5-7B",
      "partner": "Llama-3.1-8B",
      "level": "different family",
      "kappa": 0.42,
      "c": 29.4,
      "w": 1.8
    }
  ]
};
