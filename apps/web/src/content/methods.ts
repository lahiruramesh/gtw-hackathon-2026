/**
 * Learning-methods matrix: a team assessment, not a benchmark. Scores are 1 to 5 where higher is
 * always better for SKF (e.g. "Data effort 5" means little data is needed). Only the locomotion
 * column is backed by measurements from this repository (see EVIDENCE); manipulation and workflow
 * scores are judgement informed by published results. Change through a reviewed pull request.
 */

export const DOMAINS = [
  { id: "locomotion", label: "Locomotion" },
  { id: "manipulation", label: "Manipulation" },
  { id: "workflow", label: "Full workflows" },
] as const;

export type DomainId = (typeof DOMAINS)[number]["id"];

export const REQUIREMENTS = [
  { id: "data", label: "Data effort", five: "No demonstrations or labelled data needed" },
  { id: "time", label: "Training time", five: "A working skill in hours, not weeks" },
  { id: "safety", label: "Safety", five: "Behaviour can be bounded, tested and certified" },
  { id: "repeatability", label: "Repeatability", five: "Same input, same result, every shift" },
  { id: "adaptability", label: "Adaptability", five: "Copes with new parts, layouts and disturbances" },
  { id: "compute", label: "Compute", five: "Runs and trains on modest hardware" },
  { id: "explainability", label: "Explainability", five: "Engineers can say why it acted" },
  { id: "maturity", label: "Deployment maturity", five: "Proven in industrial use today" },
] as const;

export type RequirementId = (typeof REQUIREMENTS)[number]["id"];

export type Scores = Record<RequirementId, 1 | 2 | 3 | 4 | 5>;

export interface MethodAssessment {
  scores: Scores;
  /** Why the scores look like this, in one or two sentences. */
  rationale: string;
}

export interface Method {
  id: string;
  name: string;
  summary: string;
  assessments: Record<DomainId, MethodAssessment>;
}

/** Default weights (0 to 5): safety and repeatability matter most on a factory floor. */
export const DEFAULT_WEIGHTS: Record<RequirementId, number> = {
  data: 3,
  time: 3,
  safety: 5,
  repeatability: 4,
  adaptability: 3,
  compute: 2,
  explainability: 3,
  maturity: 4,
};

export const METHODS: readonly Method[] = [
  {
    id: "manual",
    name: "Manual programming",
    summary: "Hand-written controllers and motion scripts: model-based gaits, waypoints, state machines.",
    assessments: {
      locomotion: {
        scores: {
          data: 5,
          time: 2,
          safety: 3,
          repeatability: 5,
          adaptability: 2,
          compute: 5,
          explainability: 5,
          maturity: 2,
        },
        rationale:
          "Model-based humanoid gaits are analysable but take months of expert tuning and stay brittle on uneven ground (safety 3: falls off the flat). Unitree itself ships a learned policy for the G1 (maturity 2).",
      },
      manipulation: {
        scores: {
          data: 5,
          time: 2,
          safety: 5,
          repeatability: 5,
          adaptability: 1,
          compute: 5,
          explainability: 5,
          maturity: 5,
        },
        rationale:
          "How industrial arms are programmed today: certified and exact, but every new part or fixture means new programming.",
      },
      workflow: {
        scores: {
          data: 5,
          time: 2,
          safety: 5,
          repeatability: 5,
          adaptability: 1,
          compute: 5,
          explainability: 5,
          maturity: 5,
        },
        rationale:
          "PLC-style sequencing is the proven way to chain steps. It is also the right glue between learned skills.",
      },
    },
  },
  {
    id: "teleop",
    name: "Teleoperation / learning from demonstration",
    summary:
      "An operator drives the robot (VR, leader arms, mocap) and the recorded motion is replayed or generalised.",
    assessments: {
      locomotion: {
        scores: {
          data: 2,
          time: 3,
          safety: 2,
          repeatability: 3,
          adaptability: 2,
          compute: 4,
          explainability: 2,
          maturity: 2,
        },
        rationale:
          "Whole-body balance is hard to teleoperate; demonstrations do not teach recovery from pushes or slips.",
      },
      manipulation: {
        scores: {
          data: 3,
          time: 4,
          safety: 3,
          repeatability: 3,
          adaptability: 3,
          compute: 4,
          explainability: 2,
          maturity: 3,
        },
        rationale:
          "Tens to hundreds of demonstrations per task are enough for many pick-and-place skills, and process experts can provide them without ML knowledge.",
      },
      workflow: {
        scores: {
          data: 2,
          time: 3,
          safety: 3,
          repeatability: 2,
          adaptability: 2,
          compute: 4,
          explainability: 2,
          maturity: 2,
        },
        rationale: "Long demonstrations are costly and every variation of the workflow needs its own recordings.",
      },
    },
  },
  {
    id: "imitation",
    name: "Imitation learning",
    summary:
      "Policies trained on demonstration datasets (behaviour cloning, ACT, diffusion policies, motion retargeting).",
    assessments: {
      locomotion: {
        scores: {
          data: 2,
          time: 3,
          safety: 2,
          repeatability: 3,
          adaptability: 2,
          compute: 3,
          explainability: 2,
          maturity: 2,
        },
        rationale:
          "Useful for natural-looking motion styles (mocap priors), but it does not learn balance recovery without an RL stage on top.",
      },
      manipulation: {
        scores: {
          data: 2,
          time: 3,
          safety: 3,
          repeatability: 3,
          adaptability: 3,
          compute: 3,
          explainability: 2,
          maturity: 3,
        },
        rationale:
          "The strongest learned approach for contact-rich manipulation today, provided a data pipeline for demonstrations exists.",
      },
      workflow: {
        scores: {
          data: 2,
          time: 2,
          safety: 2,
          repeatability: 2,
          adaptability: 2,
          compute: 3,
          explainability: 2,
          maturity: 2,
        },
        rationale: "Errors compound over long horizons; works best when each step is a separate, gated skill.",
      },
    },
  },
  {
    id: "rl",
    name: "Reinforcement learning",
    summary:
      "Policies trained by trial and error against a reward in simulation (here: Brax PPO on MuJoCo Playground).",
    assessments: {
      locomotion: {
        scores: {
          data: 5,
          time: 4,
          safety: 3,
          repeatability: 3,
          adaptability: 4,
          compute: 3,
          explainability: 2,
          maturity: 4,
        },
        rationale:
          "Measured here: step-length control in 95 GPU-minutes and about one engineer-day, no demonstrations. Stairs needed 11 versions, so reward design is the real cost.",
      },
      manipulation: {
        scores: {
          data: 4,
          time: 2,
          safety: 2,
          repeatability: 3,
          adaptability: 3,
          compute: 2,
          explainability: 2,
          maturity: 2,
        },
        rationale:
          "Contact-rich simulation and reward shaping are slow to get right; exploration can find unsafe shortcuts.",
      },
      workflow: {
        scores: {
          data: 4,
          time: 1,
          safety: 2,
          repeatability: 2,
          adaptability: 3,
          compute: 1,
          explainability: 1,
          maturity: 1,
        },
        rationale:
          "Rewards for long multi-step tasks are sparse; training end to end is impractical at industrial reliability.",
      },
    },
  },
  {
    id: "sim2real",
    name: "Sim-to-real with domain randomisation",
    summary: "RL trained over randomised physics, delays and pushes so the policy transfers to hardware.",
    assessments: {
      locomotion: {
        scores: {
          data: 5,
          time: 3,
          safety: 4,
          repeatability: 4,
          adaptability: 4,
          compute: 3,
          explainability: 2,
          maturity: 4,
        },
        rationale:
          "Measured here (E3): randomisation cut falls under 0.5 m/s pushes from 33% to 0% but made step length about 4x less precise. Randomise what the site will really see.",
      },
      manipulation: {
        scores: {
          data: 4,
          time: 2,
          safety: 3,
          repeatability: 3,
          adaptability: 3,
          compute: 2,
          explainability: 2,
          maturity: 2,
        },
        rationale:
          "Works in research for in-hand manipulation, but contact and deformable parts remain hard to randomise faithfully.",
      },
      workflow: {
        scores: {
          data: 4,
          time: 1,
          safety: 2,
          repeatability: 2,
          adaptability: 3,
          compute: 1,
          explainability: 1,
          maturity: 1,
        },
        rationale: "Adds robustness per skill, not sequencing; costs more compute for every skill in the chain.",
      },
    },
  },
  {
    id: "feedback",
    name: "Human feedback",
    summary: "People rank, correct or approve behaviour, and the policy is refined from those signals.",
    assessments: {
      locomotion: {
        scores: {
          data: 3,
          time: 2,
          safety: 3,
          repeatability: 3,
          adaptability: 3,
          compute: 3,
          explainability: 2,
          maturity: 1,
        },
        rationale: "Rarely used for gaits; physical metrics (falls, tracking error) are cheaper judges than people.",
      },
      manipulation: {
        scores: {
          data: 3,
          time: 3,
          safety: 3,
          repeatability: 3,
          adaptability: 4,
          compute: 3,
          explainability: 2,
          maturity: 2,
        },
        rationale:
          "Corrections from operators fix the cases demonstrations missed; needs tooling to capture feedback on the floor.",
      },
      workflow: {
        scores: {
          data: 3,
          time: 2,
          safety: 3,
          repeatability: 3,
          adaptability: 4,
          compute: 3,
          explainability: 3,
          maturity: 2,
        },
        rationale:
          "Approvals and corrections at step boundaries are explicit and auditable, which suits a gated release process.",
      },
    },
  },
  {
    id: "vla",
    name: "Vision-language-action / foundation models",
    summary: "Large pretrained models that map camera images and instructions to actions, fine-tuned per task.",
    assessments: {
      locomotion: {
        scores: {
          data: 2,
          time: 2,
          safety: 1,
          repeatability: 2,
          adaptability: 3,
          compute: 1,
          explainability: 1,
          maturity: 1,
        },
        rationale: "Not where these models are strong; low-level balance still comes from a separate RL controller.",
      },
      manipulation: {
        scores: {
          data: 3,
          time: 3,
          safety: 2,
          repeatability: 2,
          adaptability: 5,
          compute: 1,
          explainability: 1,
          maturity: 2,
        },
        rationale:
          "Best generalisation to new objects and instructions, but latency, GPU cost and hard-to-bound failures keep them in pilots.",
      },
      workflow: {
        scores: {
          data: 3,
          time: 3,
          safety: 2,
          repeatability: 2,
          adaptability: 5,
          compute: 1,
          explainability: 2,
          maturity: 2,
        },
        rationale:
          "Promising as a high-level planner that sequences gated skills; its language plans can be reviewed before execution.",
      },
    },
  },
];

export interface EvidenceItem {
  title: string;
  method: string;
  facts: string[];
  takeaway: string;
  /** Where to see it in the app (imported history runs). */
  links: { label: string; href: string }[];
  /** Repository files the numbers come from. */
  sources: string[];
}

/** Measured results from this repository (docs/pipeline.md, docs/effort_log.csv). */
export const EVIDENCE: readonly EvidenceItem[] = [
  {
    title: "Step length on flat ground (v1)",
    method: "RL + domain randomisation",
    facts: [
      "202M environment steps in 95 min on one free Kaggle T4 (about 1.6 GPU-h), plus roughly one engineer-day, mostly setup and evaluation",
      "Step-length error 1.6 to 3.2 cm at step rates of 1.2 Hz and above in the training engine, no falls",
      "In an unseen engine (MuJoCo C) it still walks without falls, but the error grows to about 10 cm: the sim-to-sim gap in miniature",
    ],
    takeaway:
      "For a well-posed locomotion task, one added reward term and observation were enough. No demonstrations needed.",
    links: [{ label: "Step-length runs", href: "/runs?q=steplength" }],
    sources: ["docs/pipeline.md §3–5", "docs/effort_log.csv"],
  },
  {
    title: "E3: what domain randomisation buys",
    method: "Sim-to-real with DR vs plain RL",
    facts: [
      "Same task and budget (202M steps, about 97 min on a T4 each)",
      "Step error in the training engine: 5.9 cm with randomisation vs 1.6 cm without; in the unseen engine 10.7 vs 4.4 cm",
      "Falls under 0.5 m/s pushes: 0% with randomisation vs 33% without; under 1.0 m/s: 67% vs 100%",
      "Neither handles friction ×0.3 or 20–40 ms actuation delay, because neither was trained on them",
    ],
    takeaway:
      "Randomisation is not free: randomise exactly what the site will see (delay, pushes, measured friction) and budget training for it.",
    links: [{ label: "Compare runs", href: "/compare" }],
    sources: ["docs/pipeline.md §4c", "results/e3_comparison.md"],
  },
  {
    title: "E5: do we need learning at all?",
    method: "Vendor policy, re-parameterised",
    facts: [
      "Unitree's pretrained G1 policy, 105 runs (7 speeds × 5 gait periods × 3 seeds) in 15 s on a laptop CPU, no falls",
      "Step length from about 7 to 40 cm by re-timing the gait clock, but speed and step length stay coupled",
    ],
    takeaway:
      "Coarse changes cost nothing; precise, independent control is what training adds. Always check this baseline first.",
    links: [],
    sources: ["docs/pipeline.md §2", "results/e5/e5_runs.csv"],
  },
  {
    title: "Stair climbing, v1 to v11",
    method: "RL + curriculum + height scan + warm starts",
    facts: [
      "11 versions in about two days; 25.6 GPU-h logged across v2–v11 (v7 and v9 not logged), on Kaggle T4 and AWS L40S",
      "Held-out stairs, single start: v1 crossed 1/32 and fell on 27; v9 crossed 27/32 with 5 falls",
      "Strict 96-run test (3 starts per staircase) of the final policies, which the release gate judges: v10 78/96 crossed, 18 falls, certified 4.7 cm; v11 85/96 (89%), 11 falls, certified 6.4 cm. v9 was only strict-tested at 369.3M steps: 81/96, 9 falls, certified 4.7 cm",
      "Earlier checkpoints did better than the final policies: v10 @ 253.6M steps 82/96, 14 falls, certified 8.1 cm; v11 @ 190.7M steps 88/96 (92%), 8 falls, certified 8.1 cm",
      "Against the gate (≥ 90% crossed, ≤ 5 falls, certified ≥ 10 cm) v11's final policy meets no criterion and its 190.7M checkpoint only the crossing rate, so no version is releasable yet",
      "The biggest jumps came from fixing the simulation (one-point foot contacts, action limits capping knee torque), not from reward tuning",
    ],
    takeaway:
      "Harder skills are an iteration loop, not a single training run. Lineage, a strict test and an automatic gate are what make that loop trustworthy.",
    links: [{ label: "Stairs runs", href: "/runs?q=stairs" }],
    sources: ["docs/effort_log.csv", "results/stairs/strict_v*.json", "docs/pipeline.md §7"],
  },
  {
    title: "Compute and tooling",
    method: "Infrastructure",
    facts: [
      "Kaggle T4: about 36k steps/s once warm; free, but 30 GPU-h per week (the stairs work used up a week's quota)",
      "AWS L40S: about 100k steps/s (5× Kaggle) only after silencing the MuJoCo Warp solver, which had printed 30 GB of log in 40 min",
      "New AWS accounts start with a GPU quota of 0 and larger instances often had no capacity",
    ],
    takeaway:
      "Quota tracking, noise filtering at the source and capacity retries are product requirements, not afterthoughts.",
    links: [{ label: "Compute targets", href: "/compute" }],
    sources: ["docs/effort_log.csv"],
  },
];

export const RECOMMENDATION = [
  "Locomotion: RL with sim-to-real hardening. It ranks first with the default weights, and it is the only approach that took the G1 onto stairs in this project.",
  "Manipulation: manual programming ranks first with the default weights, and it stays the right choice wherever parts and positions are fixed, as in today's robot cells. The ranking cannot see that constraint: where parts vary, the best-ranked learned methods are the demonstration-based ones (teleoperation and imitation learning, ahead of RL), so that is what to pilot.",
  "Full workflows: classical sequencing (the manual programming row) of gated, learned skills, which also ranks first. Treat VLA models as a future high-level planner once they can be bounded and tested like any other skill.",
] as const;
