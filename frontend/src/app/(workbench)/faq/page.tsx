import { BookOpen } from "lucide-react";

export const metadata = {
  title: "PIEmaker · FAQ & Glossary",
};

interface QA {
  q: string;
  a: React.ReactNode;
}

const FAQS: QA[] = [
  {
    q: "What is PIEmaker?",
    a: (
      <>
        A campaign-level incrementality prediction workbench. Given a planned
        ad campaign, PIEmaker forecasts its <strong>ICPD</strong>{" "}
        (incremental conversions per dollar) by training a model on
        historical RCTs. Built on the methodology in NBER w35044 (Gordon,
        Moakler &amp; Zettelmeyer, April 2026).
      </>
    ),
  },
  {
    q: "What is incrementality?",
    a: (
      <>
        The conversions that happened <strong>because of</strong> the ad,
        not just <em>with</em> the ad. Platforms report attributed
        conversions which include people who would have converted anyway.
        Incrementality strips that out by comparing exposed test users to
        unexposed control users (an RCT).
      </>
    ),
  },
  {
    q: "Why are the predicted ICPDs lower than my platform's reported CPA inverse?",
    a: (
      <>
        Because platform CPAs include incidental conversions (people who
        would have converted anyway). PIEmaker reports{" "}
        <strong>incremental</strong> CPA — the true effect. The gap between
        the two is exactly the value of running this workbench.
      </>
    ),
  },
  {
    q: "What's the donor pool?",
    a: (
      <>
        The set of historical RCTs that the model learns from. Bigger,
        cleaner pools give better predictions. PIEmaker gates training by
        pool size:
        <ul className="mt-2 ml-5 list-disc space-y-1">
          <li>&lt;200 RCTs → blocked (R² ~0.37, too noisy for production)</li>
          <li>200–399 → research mode (predictions watermarked)</li>
          <li>400–1599 → production (R² 0.72–0.81)</li>
          <li>≥1600 → full production (R² approaches paper baseline 0.88)</li>
        </ul>
      </>
    ),
  },
  {
    q: "What's an RCT?",
    a: (
      <>
        Randomized Controlled Trial. A campaign that ran with a deliberate
        control group: random users were withheld from seeing the ad, so the
        difference between test-group conversions and control-group
        conversions is the campaign&rsquo;s true incremental effect.
      </>
    ),
  },
  {
    q: "What does R² ceiling mean?",
    a: (
      <>
        The theoretical upper bound on how well any model can fit the data,
        given the noise in the outcome (paper §5.2). If a model&rsquo;s R²
        equals the ceiling, you&rsquo;re at the noise floor — adding features
        can&rsquo;t help. If R² is much lower than the ceiling, the model is
        underfit and you can squeeze more signal out.
      </>
    ),
  },
  {
    q: "What's hold-out-one-level extrapolation?",
    a: (
      <>
        For each level of a segmentation variable (e.g. vertical=&ldquo;media&rdquo;), train on
        every <em>other</em> level, then score the held-out one. The R² gap
        between within-level and extrapolation tells you how much the model
        can be trusted on a campaign in that regime. Severe gaps (≥25pp) =
        block; high (≥15pp) = deprioritize. Paper Table 1.
      </>
    ),
  },
  {
    q: "Why is the simulator gated to production-only models?",
    a: (
      <>
        The Decision Simulator turns predictions into actionable budget
        shifts. Research-mode models (&lt;400 RCTs) carry too much
        uncertainty to drive money — the watermark on predictions is just an
        advisory; the simulator hard-blocks. Promote a model to production
        only after the donor pool crosses 400 admitted RCTs.
      </>
    ),
  },
  {
    q: "Why does the dashboard need so much data first?",
    a: (
      <>
        &ldquo;Trust before UX.&rdquo; You need a labeled donor pool, a trained model,
        ablation results, and hold-out diagnostics before any prediction is
        meaningful. The Phase 0–2 surfaces (donor pool, labels, features,
        model trust) build that foundation. Phase 3 (predict, portfolio,
        decisions) and Phase 4 (drift, simulator) consume it.
      </>
    ),
  },
];

const GLOSSARY: { term: string; def: React.ReactNode }[] = [
  { term: "ATT", def: "Average Treatment Effect on the Treated. Eq. 14 in the paper." },
  { term: "IC", def: "Incremental Conversions. Eq. 22: ATT × exposure_rate × test_users." },
  { term: "ICPD", def: "Incremental Conversions Per Dollar. Eq. 23: IC / cost. The model's prediction target." },
  { term: "LCC", def: "Last-Click Conversion. The naive attribution baseline. LCC-7d/$ is one of the strongest features in the model." },
  { term: "Exposure rate (D̄)", def: "Fraction of test users who actually saw the ad. Critical for ATT computation." },
  { term: "X_pre", def: "Pre-determined features — knowable before launch (objective, vertical, audience, planned spend tier)." },
  { term: "X_post", def: "Post-determined features — only knowable after the campaign runs (CTR, exposure rate, LCC-7d/$, conversions/$)." },
  { term: "MC defense", def: "Mechanical-Correlation defense. Decides per-row whether to use sample_split (gold standard, requires user-level data), shared_sample_compromise (acceptable for large test pools), or block. Paper §4.4." },
  { term: "Sample split", def: "Deterministic test/control split that prevents the same user from appearing in both arms. Requires user-level data." },
  { term: "Shared-sample compromise", def: "Fallback when user-level data isn't available. Shares the test pool across runs but bounds bias for large pools." },
  { term: "Hold-out-one-level", def: "Cross-validation that trains on every level of a segmentation variable except one, then scores the held-out level. Surfaces extrapolation risk per regime. Paper Table 1." },
  { term: "PSI", def: "Population Stability Index. Measures distribution drift between training and scoring. Standard bands: stable <0.10, moderate 0.10-0.25, severe ≥0.25." },
  { term: "Bootstrap CI", def: "95% confidence interval on the model's R², computed by resampling the training set N times. Conveys how stable the goodness-of-fit number is." },
  { term: "R² ceiling", def: "Upper bound on R² given outcome noise. Paper §5.2." },
  { term: "Ablation", def: "Train multiple model variants with progressively more features (Pre → Pre+Yt → Pre+Yt+LCC-7D → Full → Raw_LCC). Paper Figure 2." },
  { term: "Risk-adjusted ICPD", def: "ICPD or its CI lower bound multiplied by a risk multiplier (severe -inf, high 0.5, medium 0.85, low/unknown 1.0). Used to rank campaigns in the Decisions page." },
  { term: "Cap multiplier", def: "Bound in the Decision Simulator: no campaign can grow beyond cap × original_spend after reallocation. Cap=1.0 = status-quo." },
];

export default function FaqPage() {
  return (
    <main className="container py-12">
      <header className="mb-6 flex items-start gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-md bg-primary/10 text-primary">
          <BookOpen className="h-5 w-5" />
        </span>
        <div>
          <p className="text-sm uppercase tracking-widest text-muted-foreground">
            Reference
          </p>
          <h1 className="mt-1 text-3xl font-semibold">FAQ &amp; Glossary</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Methodology, terminology, and the questions newcomers ask.
          </p>
        </div>
      </header>

      <section className="mb-12">
        <h2 className="mb-4 text-xl font-medium">Frequently asked questions</h2>
        <div className="space-y-3">
          {FAQS.map((qa, i) => (
            <details
              key={i}
              className="group rounded-lg border bg-card p-5 shadow-sm open:shadow-md"
            >
              <summary className="cursor-pointer font-medium marker:text-muted-foreground">
                {qa.q}
              </summary>
              <div className="mt-3 text-sm leading-relaxed text-muted-foreground">
                {qa.a}
              </div>
            </details>
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-4 text-xl font-medium">Glossary</h2>
        <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-secondary">
              <tr>
                <th className="w-48 px-4 py-2.5 text-left font-medium">Term</th>
                <th className="px-4 py-2.5 text-left font-medium">Definition</th>
              </tr>
            </thead>
            <tbody>
              {GLOSSARY.map((g) => (
                <tr key={g.term} className="border-t">
                  <td className="px-4 py-3 align-top font-mono text-xs">
                    {g.term}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">{g.def}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="mt-10 text-xs text-muted-foreground">
        <p>
          Built on Gordon, Moakler &amp; Zettelmeyer,{" "}
          <em>
            &ldquo;Predicted Incrementality from Experiments: A Method to Forecast
            Ad Effectiveness&rdquo;
          </em>
          , NBER Working Paper w35044 (April 2026).
        </p>
        <p className="mt-1">© 2026 PIEmaker · Built by Prima Hanura Akbar</p>
      </footer>
    </main>
  );
}
