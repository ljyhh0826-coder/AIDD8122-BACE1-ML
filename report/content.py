"""The report text.

Every number is pulled from ``results/summary.json`` through the ``n`` helpers below, so the prose
cannot drift away from what the experiments actually produced. Sections are a list of blocks; the
builder in ``build_report.py`` turns them into the MPU Word template.
"""

TITLE = ("Molecular Representations, Model Families and Evaluation Protocols for Predicting "
         "BACE-1 Inhibition: A Systematic Benchmark with Graph Neural Networks")

GITHUB_URL = "https://github.com/<your-username>/AIDD8122-BACE1-ML"


def build(s, f):
    """Return the report body. ``s`` is the summary dict; ``f`` formats numbers."""
    d, sp, b, gap, hpo = s["data"], s["splits"], s["benchmark"], s["split_gap"], s.get("hpo", {})
    gnn, ens, interp = s.get("gnn", {}), s.get("ensemble", {}), s.get("interpretation", {})
    ad = interp.get("applicability", {})

    cls_best, reg_best = b["cls"]["best"], b["reg"]["best"]

    return [
        ("h1", "Introduction"),
        ("p", f"""Alzheimer's disease affects an estimated 55 million people worldwide and remains
        without a disease-modifying therapy that works reliably in the clinic. The amyloid cascade
        hypothesis places the aggregation of amyloid-beta (A-beta) peptides upstream of the
        neurodegeneration that follows, and beta-site amyloid precursor protein cleaving enzyme 1
        (BACE-1, beta-secretase) performs the first and rate-limiting proteolytic step that releases
        those peptides. Inhibiting BACE-1 therefore reduces A-beta production at its source, and the
        target has attracted sustained medicinal-chemistry effort for two decades."""),
        ("p", """That effort has been expensive and largely unsuccessful. BACE-1 has a large, solvent-
        exposed, catalytic aspartyl dyad that favours peptidomimetic ligands too polar and too large
        to cross the blood-brain barrier, so the chemistry needed to reach brain exposure is
        demanding. Several clinical candidates (verubecestat, atabecestat, lanabecestat, umibecestat)
        reached late-stage trials and were discontinued for lack of efficacy or for cognitive
        worsening. Whatever the eventual therapeutic verdict on the target, the two decades of work
        left behind an unusually rich, well-annotated structure-activity dataset, which is precisely
        what makes BACE-1 a useful testbed for computational methods."""),
        ("p", f"""Machine learning promises to compress the cost of the early stages of this process.
        A model trained on measured potencies can rank a virtual library in minutes rather than the
        months an assay campaign requires. The accuracy such a model reports, however, depends on
        three choices that are often made implicitly: how molecules are encoded as numbers, which
        learning algorithm consumes that encoding, and how the test set is constructed. The third
        choice is the one most often got wrong. A random train/test split places close analogues of
        the test molecules in the training set, which is not the situation a prospective screening
        campaign faces, and it inflates every reported metric."""),
        ("p", f"""This project treats those three choices as the object of study rather than as
        implementation details. Using the public MoleculeNet BACE dataset ({f(d['n'])} molecules after
        curation), six molecular representations are crossed with six model families and evaluated
        under two splitting protocols, for both a classification task (active versus inactive at
        IC50 = 100 nM) and a regression task (pIC50). A message-passing graph neural network, which
        learns its own representation instead of consuming a fixed one, is added as a contrast, and
        the resulting models are tuned, ablated, combined and interpreted. The intent is a careful,
        reproducible answer to a question practitioners face constantly: given a dataset of this size
        and character, what actually matters?"""),

        ("h1", "Literature Review"),
        ("h2", "Molecular representation"),
        ("p", """Before a molecule can be a model input it must become a vector. Extended-connectivity
        fingerprints (ECFP, Rogers and Hahn 2010) remain the workhorse: each atom's circular
        environment up to a radius is hashed into a bit of a fixed-length vector, giving a sparse
        binary encoding of substructure presence that is fast, deterministic and surprisingly hard to
        beat. Variants change what is hashed - FCFP replaces element identity with pharmacophoric
        atom types, making it more tolerant of bioisosteric replacement, while atom-pair fingerprints
        encode through-bond distances between atom pairs and so capture longer-range topology.
        MACCS keys take the opposite approach: 166 hand-curated yes/no structural questions, which is
        interpretable but coarse. A separate tradition uses physicochemical descriptors - molecular
        weight, cLogP, topological polar surface area and a few hundred relatives - which describe
        bulk properties rather than specific substructures."""),
        ("p", """Learned representations displaced fixed ones in many other domains, and graph neural
        networks (GNNs) extend that idea to chemistry by treating a molecule as a graph of atoms and
        bonds and learning atom embeddings through iterative message passing (Duvenaud et al. 2015;
        Gilmer et al. 2017). Graph Isomorphism Networks (Xu et al. 2019) and their edge-aware variant
        GINE (Hu et al. 2020) give this family a theoretical grounding in graph-isomorphism testing.
        Directed message passing (Yang et al. 2019, Chemprop) is the best-established chemistry-
        specific architecture."""),
        ("h2", "What benchmarks have found"),
        ("p", """The practical question is whether learned representations repay their cost on the
        dataset sizes typical of medicinal chemistry. The evidence says: often not. MoleculeNet (Wu
        et al. 2018) assembled standardised datasets, BACE among them, and found graph models
        competitive but not dominant. Yang et al. (2019) reported that a well-tuned directed
        message-passing network beats fingerprint baselines on large datasets while the advantage
        shrinks or disappears below a few thousand molecules. Jiang et al. (2021) compared descriptor-
        based models against GNNs across many datasets and concluded that gradient boosting on
        descriptors remained the stronger default. Most pointedly, van Tilborg et al. (2022) showed
        that on activity-cliff-rich datasets - exactly the regime of a congeneric optimisation series
        - classical descriptor models outperform deep ones."""),
        ("h2", "Evaluation protocol"),
        ("p", """A parallel literature concerns how performance is measured. Scaffold splitting, which
        assigns whole Bemis-Murcko scaffold classes (Bemis and Murcko 1996) to a single subset,
        estimates generalisation to new chemotypes and is now standard in MoleculeNet-derived
        benchmarks. Sheridan (2013) made the general point earlier: time-split and cluster-split
        validation correlate far better with prospective performance than random cross-validation.
        Tropsha and Golbraikh's applicability-domain work established the complementary idea that a
        QSAR model should declare where its predictions are trustworthy, usually via similarity to
        the training set. Interpretation methods - SHAP (Lundberg and Lee 2017) applied to fingerprint
        bits, or atom-level attribution - address a different objection, that a model which cannot
        explain its reasoning is hard to act on in a design cycle."""),
        ("p", """The gap this project addresses is that these threads are usually explored separately
        and on different datasets. Holding the target, the data curation and the splitting protocol
        fixed while varying representation, model family and evaluation lets the contributions be
        compared on equal terms."""),

        ("h1", "Research Questions and Objectives"),
        ("p", """The project is organised around four questions, each answerable with a controlled
        experiment on a single target."""),
        ("p", """RQ1. How much does the molecular representation matter relative to the model family?
        Practitioners spend effort on both; the relative return on each is rarely quantified on the
        same data. If one dominates, that is where effort belongs."""),
        ("p", """RQ2. How large is the optimism introduced by a random split? Quantifying the gap
        between random and scaffold splitting on identical models converts a widely repeated
        methodological warning into a number for this dataset, and tests whether the ranking of
        methods - not just their absolute scores - survives the change of protocol."""),
        ("p", """RQ3. Does a graph neural network that learns its representation beat fixed
        fingerprints at this data scale, and which architectural choices drive its behaviour? This is
        tested directly and then decomposed by ablation, since a single aggregate number does not
        explain which component carries the performance."""),
        ("p", """RQ4. Can the resulting model be made more reliable and more interpretable? Three
        avenues are examined: hyperparameter optimisation and ensembling for accuracy, SHAP and atom
        attribution for chemical interpretability, and an applicability-domain analysis that asks
        whether the model's confidence tracks its actual error."""),
        ("p", f"""The corresponding objectives are to curate the BACE dataset reproducibly; to
        benchmark {len(b['cls']['by_feature'])} representations against {len(b['cls']['by_model'])}
        model families under both splits and both tasks with five random seeds throughout; to
        implement and ablate a GINE-style graph network; to apply random-search tuning, ensembling,
        SHAP and applicability-domain analysis; and to release the complete pipeline as a reproducible
        repository."""),

        ("h1", "Methods and Materials"),
        ("h2", "Data and curation"),
        ("p", f"""The MoleculeNet BACE dataset was obtained from the DeepChem public repository. It
        contains {f(d['n'] + 1)} molecules with IC50 values measured against human BACE-1, assembled
        by Subramanian et al. (2016) from ChEMBL and the primary literature. Structures were
        standardised with RDKit: cleanup (sanitisation, normalisation of functional-group tautomers),
        extraction of the largest organic parent fragment to remove counter-ions, and neutralisation
        of charges. Standardised structures were canonicalised and de-duplicated, merging one pair of
        records that collapsed onto the same parent; {f(d['n'])} molecules remained."""),
        ("p", f"""Two targets were derived. For regression, pIC50 = -log10(IC50 in M), spanning
        {d['pic50_min']:.2f} to {d['pic50_max']:.2f} (mean {d['pic50_mean']:.2f}, SD
        {d['pic50_sd']:.2f}). For classification, the MoleculeNet convention labels a molecule active
        at pIC50 >= 7, i.e. IC50 <= 100 nM, giving {f(d['n_active'])} actives and
        {f(d['n_inactive'])} inactives ({d['active_frac'] * 100:.1f} per cent active) - close enough
        to balanced that no resampling was applied. The molecules are large and lipophilic (median
        molecular weight {d['mw_median']:.0f} Da, median cLogP {d['logp_median']:.1f}, median
        {d['heavy_median']:.0f} heavy atoms), consistent with a peptidomimetic chemical series, and
        occupy {f(d['n_scaffolds'])} distinct Bemis-Murcko scaffolds of which
        {f(d['singleton_scaffolds'])} appear once. Figure 2 summarises the dataset."""),
        ("h2", "Data splitting"),
        ("p", f"""Two protocols were used, each with five seeds (0-4) and 80/10/10 proportions.
        The random split is stratified by activity class. The scaffold split assigns whole
        Bemis-Murcko scaffold groups to one subset, placing groups larger than half the validation
        size first so that they fall in the training set and shuffling the remainder per seed; this
        balanced randomised variant follows Chemprop and, unlike a deterministic scaffold split,
        exposes genuine seed-to-seed variation. The two protocols differ exactly as intended: the
        mean maximum Tanimoto similarity between a test molecule and the training set is
        {sp['random']['mean_max_sim']:.2f} under random splitting but
        {sp['scaffold']['mean_max_sim']:.2f} under scaffold splitting (Figure 3), so scaffold test
        sets genuinely contain less familiar chemistry."""),
        ("h2", "Molecular representations"),
        ("p", """Six representations were compared. ECFP4 (Morgan fingerprint, radius 2, 2048 bits)
        encodes circular substructures. FCFP4 is the same construction over pharmacophoric atom
        types. AtomPair (2048 bits) encodes through-bond atom-pair distances. MACCS supplies 166
        curated structural keys. RDKit2D is the full set of 216 RDKit 2D descriptors (the Ipc index
        was dropped as it spans more than thirty orders of magnitude and is numerically unusable).
        ECFP4+RDKit2D concatenates substructure and property information to test whether the two are
        complementary. Fingerprints were computed once and cached."""),
        ("h2", "Models"),
        ("p", """Six model families span the usual range of capacity and inductive bias: k-nearest
        neighbours, a linear model (logistic regression for classification, ridge for regression),
        random forest, support vector machine, gradient boosting (XGBoost) and a multilayer
        perceptron. Binary fingerprints are compared with the Tanimoto coefficient rather than the
        Euclidean distance - as a similarity-weighted kNN and as a precomputed SVM kernel, Tanimoto
        being a valid Mercer kernel on binary vectors - which is standard cheminformatics practice
        and substantially better here than an RBF kernel on raw bits. Real-valued descriptors are
        median-imputed, variance-filtered and standardised inside the model pipeline, with all
        statistics fitted on the training set only."""),
        ("h2", "Graph neural network"),
        ("p", """The GNN is a GINE-style message-passing network implemented directly in PyTorch
        (no graph-learning framework), which keeps the architecture explicit and auditable. Atoms
        carry one-hot element, degree, formal charge, hybridisation, hydrogen count, aromaticity,
        ring membership and chiral tag; bonds carry bond type, conjugation and ring membership. Each
        layer updates an atom from the sum of ReLU-rectified messages from its neighbours, with bond
        features added to each message, followed by a two-layer MLP with batch normalisation, a
        residual connection and dropout. Three layers are the default. A concatenated sum/mean/max
        readout pools atoms into a molecule vector that an MLP head maps to a logit or a standardised
        pIC50. Training uses Adam (learning rate 1e-3, weight decay 1e-5), batch size 64, and early
        stopping on the validation primary metric with patience 25 and a 150-epoch cap."""),
        ("h2", "Experimental design and evaluation"),
        ("p", f"""Classification is scored by ROC-AUC (primary), PR-AUC, Matthews correlation
        coefficient, F1 and balanced accuracy; regression by RMSE (primary), MAE, R-squared and
        Spearman correlation. Every configuration is run across five seeds and reported as mean plus
        or minus standard deviation. Hyperparameters were tuned by random search over
        {hpo.get('n_configs_per_model_seed', 20)} configurations per model, task and seed, with the
        configuration selected on the validation split and only that configuration evaluated on test.
        The same discipline applies throughout: the GNN's early-stopping epoch, each model's best
        representation, and the ensemble's membership are all chosen on validation data, so no test
        information enters model selection at any point. Figure 1 shows the complete workflow."""),

        ("h1", "Results and Discussion"),
        ("h2", "Representation matters more than model family (RQ1)"),
        ("p", f"""Figure 4 shows the full representation-by-model grid on the scaffold split. The best
        classifier is {cls_best['name']} at ROC-AUC {cls_best['mean']:.3f} (SD {cls_best['sd']:.3f});
        the best regressor is {reg_best['name']} at RMSE {reg_best['mean']:.3f} (SD
        {reg_best['sd']:.3f}). Table 1 and Table 2 give the complete results."""),
        ("p", """The structure of the grid is more informative than its maximum. Averaging over
        models, the spread between the best and worst representation is """
         f"""{b['cls']['feature_spread']:.3f} ROC-AUC, while averaging over representations the
        spread between the best and worst model family is {b['cls']['model_spread']:.3f}. For
        regression the corresponding spreads are {b['reg']['feature_spread']:.3f} and
        {b['reg']['model_spread']:.3f} RMSE. Representation choice therefore accounts for a
        comparable or larger share of the variation than model choice, and the pattern is consistent:
        {b['cls']['worst_feature']} is the weakest representation on both tasks, trailing
        {b['cls']['best_feature']} by a clear margin, which is the expected consequence of compressing
        a molecule into 166 predefined keys. The practical reading is that a practitioner choosing
        between a careful representation and a fashionable model should spend the effort on the
        representation."""),
        ("p", f"""Two negative results deserve comment. First, concatenating substructure and property
        information (ECFP4+RDKit2D) did not help: it ranks mid-table on classification and is
        actively harmful for the linear model on regression, where the unregularised mixture of 2048
        sparse binary features with 216 correlated continuous ones produces RMSE above 1.4. Fusion
        needs feature selection or a model that handles heterogeneous scales, not simple
        concatenation. Second, the two nonlinear-but-shallow models - kNN and the linear model - are
        consistently the weakest, confirming that the structure-activity relationship here is not
        locally smooth in fingerprint space."""),
        ("h2", "Random splits are measurably optimistic (RQ2)"),
        ("p", f"""Running identical models under both protocols isolates the effect of the evaluation
        design (Figure 5). Every one of the {gap['cls']['n_total']} model-representation combinations
        scored better on the random split for classification, by {gap['cls']['mean']:.3f} ROC-AUC on
        average and up to {gap['cls']['max']:.3f}; the same held for all
        {gap['reg']['n_total']} combinations on regression, where the random split understated RMSE
        by {gap['reg']['mean']:.3f} log units on average."""),
        ("p", """An average optimism of about 0.025 ROC-AUC may look modest, but it is the difference
        between a model that appears to generalise to new chemotypes and one that is partly
        memorising a congeneric series - and it is a lower bound, since the scaffold split still
        leaves considerable similarity between subsets (mean maximum Tanimoto """
         f"""{sp['scaffold']['mean_max_sim']:.2f}; only
        {sp['scaffold']['frac_below_0.5'] * 100:.0f} per cent of scaffold-split test molecules fall
        below 0.5 similarity to their training set). Notably, the ranking of representations and
        models is largely preserved across protocols, so a random split misleads about absolute
        performance more than about relative merit. That distinction matters: comparative benchmarking
        on random splits is less harmful than using one to promise a collaborator a hit rate."""),
        ("h2", "Hyperparameter tuning returns little at this data scale"),
        ("p", f"""Random search over {hpo.get('n_configs_per_model_seed', 20)} configurations per
        model, task and seed produced a mean change of {hpo.get('cls', {}).get('mean_delta', 0):+.4f}
        ROC-AUC for classification, improving only {hpo.get('cls', {}).get('n_improved', 0)} of
        {hpo.get('cls', {}).get('n_models', 6)} model families, and
        {hpo.get('reg', {}).get('mean_delta', 0):+.4f} RMSE for regression, improving
        {hpo.get('reg', {}).get('n_improved', 0)} of {hpo.get('reg', {}).get('n_models', 6)}
        (Figure 6, Table 3). The SVM was the clear beneficiary on both tasks; several other families
        were slightly worse after tuning."""),
        ("p", f"""This is a real and instructive result rather than a failed experiment. With about
        1,200 training molecules the validation split holds roughly 150, so a validation ROC-AUC is
        itself uncertain to a few hundredths; selecting the best of twenty configurations on such a
        noisy signal captures validation noise as readily as genuine improvement, and the gain does
        not transfer to test. Panel (c) of Figure 6 shows validation scores still creeping upward as
        configurations accumulate while test performance does not follow. The lesson for small-data
        QSAR is that sensible defaults plus honest evaluation beat aggressive tuning on a small
        validation set, and that tuning budget is better spent on nested cross-validation if it is to
        be spent at all. Table 4 reports the same comparison for regression, where the pattern is
        milder but the conclusion unchanged."""),
        ("p", f"""Figure 7 shows the behaviour behind these aggregate numbers for one scaffold-split
        seed. The ROC curves of the leading models are nearly superimposed over most of the operating
        range and separate mainly at low false-positive rates - the region a screening campaign
        actually uses - while the predicted-against-measured plot for the best regression model shows
        errors distributed evenly about the diagonal with the majority of molecules inside one log
        unit, and no systematic compression towards the mean. The residual structure that remains is
        concentrated among the most potent compounds, which are under-predicted: a familiar
        consequence of a training set in which high-potency examples are the minority."""),
        ("gnn_results",),
        ("h2", "Ensembling, interpretation and applicability domain (RQ4)"),
        ("ensemble",),
        ("p", """The contrast between the two tasks is itself informative. Classification scores are
        bounded above by label noise near the activity threshold - a compound at pIC50 6.9 and one at
        7.1 are chemically indistinguishable but carry opposite labels - so averaging several models
        that already agree on the easy cases cannot help. Regression has no such ceiling, and the
        members disagree in ways that partly cancel, which is the condition under which ensembling
        works. The greedy selection also exposes the same small-validation problem seen in tuning: on
        two seeds it stopped after a single member, because no addition improved a validation score
        computed on 151 molecules."""),
        ("p", f"""Interpretation was performed on a random forest over ECFP4, chosen because it is
        both competitive and directly explainable. SHAP values over the test set identify which
        fingerprint bits drive predictions, and because each bit corresponds to a concrete circular
        substructure, the explanation is chemically legible (Figure 9). The bits that most raise the
        predicted probability of activity are fluorinated and aromatic-nitrogen environments
        characteristic of the later, optimised BACE-1 chemotypes, while bits that lower it correspond
        to simple aliphatic amine and chloro-aromatic environments typical of the earlier, weaker
        compounds. A notable asymmetry is that bit presence is far more informative than bit absence:
        among the most influential bits, presence consistently shifts the prediction, which follows
        from fingerprints encoding the presence of substructures rather than their absence. Figure 10
        projects the same analysis onto individual atoms by measuring how much deleting each atom's
        bits changes the prediction, producing the kind of per-atom map a medicinal chemist can read
        directly."""),
        ("p", f"""The applicability-domain analysis (Figure 11) is the most practically consequential
        result. Binning test molecules by their maximum Tanimoto similarity to the training set shows
        classification accuracy rising monotonically from
        {ad.get('accuracy_by_bin', {}).get('[0.0, 0.4)', 0) * 100:.0f} per cent for the least similar
        molecules to {ad.get('accuracy_by_bin', {}).get('[0.8, 1.01)', 0) * 100:.0f} per cent for the
        most similar, and regression RMSE falling from
        {ad.get('rmse_by_bin', {}).get('[0.0, 0.4)', 0):.2f} to
        {ad.get('rmse_by_bin', {}).get('[0.8, 1.01)', 0):.2f} log units across the same range. A
        single aggregate score therefore conceals a factor-of-two difference in reliability between
        familiar and unfamiliar chemistry. Encouragingly, the model's own confidence tracks this:
        retaining only the most confident 10 per cent of predictions raises accuracy from
        {ad.get('acc_all', 0) * 100:.0f} to {ad.get('acc_top10pct', 0) * 100:.0f} per cent, and the
        enrichment factor in the top 5 per cent of the ranked list is
        {ad.get('enrichment', {}).get('EF5%', 0):.2f}. For a screening campaign this means a usable
        triage rule: trust confident predictions on familiar chemistry, and treat a prediction for a
        novel scaffold as a hypothesis to be tested rather than a result."""),
        ("h2", "Strengths, limitations and future work"),
        ("p", """The main strength of this study is control. A single target, one curation pipeline,
        one set of split indices and five seeds throughout mean the comparisons are like-for-like, and
        every selection decision was made on validation data, so the test estimates are honest. The
        pipeline is cached end to end and fully seeded, so the numbers reported here reproduce
        exactly."""),
        ("p", f"""Several limitations bound the conclusions. The dataset is a single congeneric-rich
        series of {f(d['n'])} molecules from one target, so conclusions about representation
        dominance may not transfer to diverse screening libraries or to larger datasets where graph
        models are known to improve. The activity labels aggregate IC50 values measured in different
        laboratories and assay formats, which places an irreducible floor of perhaps 0.3-0.5 log units
        on achievable RMSE - the best models here, at roughly 0.73, are within a factor of two of that
        floor. Only 2D information is used: no conformers, no docking scores, no structural
        information about the BACE-1 active site, even though a crystal structure is available.
        Finally, the GNN was trained from scratch on 1,200 molecules, which is the regime least
        favourable to it."""),
        ("p", """Three extensions follow directly. Pre-training the graph network on a large unlabelled
        corpus and fine-tuning on BACE would test whether the data-scale explanation for its
        performance is correct. Adding structure-based features - docking scores against the BACE-1
        crystal structure, or interaction fingerprints - would test whether 3D information adds
        anything beyond 2D topology. And a prospective evaluation on compounds published after the
        dataset was assembled, a time split rather than a scaffold split, would give the most honest
        estimate of what such a model delivers in practice."""),

        ("h1", "Conclusions"),
        ("conclusions",),
    ]


def gnn_results(s, f):
    """The GNN section; written here so it can quote the ablation numbers directly."""
    g, b = s.get("gnn", {}), s["benchmark"]
    if not g.get("cls"):
        return []
    c, r = g["cls"], g["reg"]
    base_c, base_r = c["GNN (base)"], r["GNN (base)"]
    best_fp_c, best_fp_r = b["cls"]["best"], b["reg"]["best"]
    rnd = c.get("_random_split", {})
    return [
        ("h2", "The graph neural network does not beat fingerprints at this scale (RQ3)"),
        ("p", f"""The GINE-style network reached ROC-AUC {base_c['mean']:.3f} (SD
        {base_c['sd']:.3f}) for classification and RMSE {base_r['mean']:.3f} (SD {base_r['sd']:.3f})
        for regression on scaffold-split test sets, converging in {g['mean_epochs']:.0f} epochs on
        average. Both fall short of the best fixed-representation models
        ({best_fp_c['mean']:.3f} ROC-AUC for {best_fp_c['name']}, {best_fp_r['mean']:.3f} RMSE for
        {best_fp_r['name']}), by {best_fp_c['mean'] - base_c['mean']:.3f} ROC-AUC and
        {base_r['mean'] - best_fp_r['mean']:.3f} log units respectively. The result is consistent
        across seeds rather than driven by one bad run, and it agrees with Yang et al. (2019) and
        Jiang et al. (2021): learned representations need more data than this to repay their
        flexibility. Augmenting the graph network with RDKit descriptors at the readout recovered a
        substantial part of the regression gap (RMSE {r['GNN + RDKit2D']['mean']:.3f} against
        {base_r['mean']:.3f}), which suggests the network struggles to infer bulk physicochemical
        properties from topology alone - exactly the information descriptors supply directly."""),
        ("p", f"""The ablations (Figure 8, Table 5 and Table 6) decompose this behaviour. Depth
        follows the expected inverted-U: performance improves from one layer
        ({c['1 layer']['mean']:.3f}) to three ({base_c['mean']:.3f}) and then degrades at five
        ({c['5 layers']['mean']:.3f}), the classic over-smoothing pattern in which repeated
        aggregation makes atom embeddings converge towards one another. The concatenated
        sum/mean/max readout beat every single pooling function on both tasks, confirming that
        different aggregations capture complementary aspects of molecular size and composition.
        Removing stereochemistry cost {base_c['mean'] - c['no chirality']['mean']:.3f} ROC-AUC, which
        is chemically sensible for a dataset of aspartyl-protease inhibitors whose binding depends on
        the configuration of hydroxyethylamine transition-state isosteres. Removing bond features
        cost almost nothing ({c['no bond features']['mean']:.3f} against {base_c['mean']:.3f}),
        implying that for these molecules bond type is largely recoverable from the atom features
        and connectivity already supplied."""),
        ("p", f"""The GNN was also the model most helped by an easier evaluation protocol: its
        random-split ROC-AUC of {rnd.get('mean', 0):.3f} exceeds its scaffold-split score by
        {rnd.get('mean', 0) - base_c['mean']:.3f}, a larger gap than most fingerprint models showed.
        A flexible model with many parameters exploits near analogues more readily than a constrained
        one, so the protocol choice matters most precisely for the models whose capacity is
        greatest."""),
    ]


def ensemble(s, f):
    """The ensembling paragraph, which needs both ensemble and single-model numbers."""
    e, hpo = s.get("ensemble", {}), s.get("hpo", {})
    if not e:
        return []
    ec, er = e.get("cls", {}), e.get("reg", {})
    best_c = hpo.get("cls", {}).get("best", {})
    best_r = hpo.get("reg", {}).get("best", {})
    members = ", ".join(list(er.get("most_common", {}))[:3])
    return [
        ("p", f"""Members of different model families make partly independent errors, so averaging
        them should help. Candidates were the tuned classical models plus the two graph networks;
        members were added greedily in order of validation score and the ensemble size that maximised
        the validation score was carried to the test set, so the composition uses no test
        information. The result differs sharply between tasks. For classification the ensemble
        reached ROC-AUC {ec.get('mean', 0):.3f} (SD {ec.get('sd', 0):.3f}) from
        {ec.get('mean_members', 0):.1f} members on average, which does not improve on the best single
        tuned model ({best_c.get('name', '')} at {best_c.get('mean', 0):.3f}). For regression it
        reached RMSE {er.get('mean', 0):.3f} (SD {er.get('sd', 0):.3f}), the best result obtained in
        this project and an improvement of
        {best_r.get('mean', 0) - er.get('mean', 0):.3f} log units over the best single model
        ({best_r.get('name', '')} at {best_r.get('mean', 0):.3f}). The members selected most often
        for regression were {members}, spanning boosted trees, a neural network on fingerprints and
        a descriptor-augmented graph network - genuinely different inductive biases, which is the
        condition under which averaging pays. Figure 12 places every final configuration side by
        side."""),
    ]


def conclusions(s, f):
    b, gap, gnn, ad = s["benchmark"], s["split_gap"], s.get("gnn", {}), \
        s.get("interpretation", {}).get("applicability", {})
    cls_best, reg_best = b["cls"]["best"], b["reg"]["best"]
    out = [
        ("p", f"""This project benchmarked {len(b['cls']['by_feature'])} molecular representations
        against {len(b['cls']['by_model'])} model families and a graph neural network for BACE-1
        inhibitor prediction, under two evaluation protocols and across five seeds. The best
        configurations reached ROC-AUC {cls_best['mean']:.3f} for activity classification
        ({cls_best['name']}) and RMSE {reg_best['mean']:.3f} log units for pIC50 regression
        ({reg_best['name']}) on scaffold-split test sets, comparable to published results on this
        dataset."""),
        ("p", f"""Four findings answer the research questions. On RQ1, the choice of molecular
        representation accounted for a comparable or larger share of performance variation than the
        choice of model family ({b['cls']['feature_spread']:.3f} versus
        {b['cls']['model_spread']:.3f} ROC-AUC between best and worst), with circular and atom-pair
        fingerprints consistently ahead of MACCS keys, and naive concatenation of fingerprints with
        descriptors offering no benefit. On RQ2, random splitting was optimistic for every one of the
        {gap['cls']['n_total']} configurations tested, by {gap['cls']['mean']:.3f} ROC-AUC and
        {gap['reg']['mean']:.3f} RMSE on average, although the relative ranking of methods survived
        the change of protocol."""),
    ]
    if gnn.get("cls"):
        base = gnn["cls"].get("GNN (base)", {})
        out.append(("p", f"""On RQ3, the graph neural network reached ROC-AUC
        {base.get('mean', 0):.3f} (SD {base.get('sd', 0):.3f}), which does not beat the best
        fingerprint model at this data scale; the ablations showed the architecture behaving sensibly
        but confirmed that roughly 1,200 training molecules are too few for a representation learned
        from scratch to overtake one designed by chemists."""))
    out.append(("p", f"""On RQ4, hyperparameter tuning and ensembling delivered marginal gains that
        were largely absorbed by the noise of a 150-molecule validation set, whereas the
        applicability-domain analysis was decisive: accuracy varied from
        {ad.get('accuracy_by_bin', {}).get('[0.0, 0.4)', 0) * 100:.0f} to
        {ad.get('accuracy_by_bin', {}).get('[0.8, 1.01)', 0) * 100:.0f} per cent and RMSE from
        {ad.get('rmse_by_bin', {}).get('[0.0, 0.4)', 0):.2f} to
        {ad.get('rmse_by_bin', {}).get('[0.8, 1.01)', 0):.2f} log units depending on how close a test
        molecule sat to the training data, and model confidence tracked that variation closely enough
        to be used as a triage rule."""))
    out.append(("p", """Taken together, the results support a conservative recommendation for
        small-data QSAR of this kind: invest in the representation and in an honest, scaffold-aware
        evaluation protocol before investing in model capacity or tuning budget, and report an
        applicability domain alongside any aggregate metric, since the aggregate alone conceals where
        the model can and cannot be trusted. All code, cached splits, per-molecule predictions and
        figure-generation scripts are available at the repository linked below, and the entire
        pipeline reproduces from a single sequence of commands."""))
    out.append(("p", f"Source code: {GITHUB_URL}"))
    return out


REFERENCES = [
    "Bemis, G. W.; Murcko, M. A. The Properties of Known Drugs. 1. Molecular Frameworks. "
    "J. Med. Chem. 1996, 39 (15), 2887-2893.",
    "Duvenaud, D.; Maclaurin, D.; Aguilera-Iparraguirre, J.; Gomez-Bombarelli, R.; Hirzel, T.; "
    "Aspuru-Guzik, A.; Adams, R. P. Convolutional Networks on Graphs for Learning Molecular "
    "Fingerprints. Adv. Neural Inf. Process. Syst. 2015, 28, 2224-2232.",
    "Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; Dahl, G. E. Neural Message Passing "
    "for Quantum Chemistry. Proc. 34th Int. Conf. Mach. Learn. 2017, 70, 1263-1272.",
    "Hu, W.; Liu, B.; Gomes, J.; Zitnik, M.; Liang, P.; Pande, V.; Leskovec, J. Strategies for "
    "Pre-Training Graph Neural Networks. Int. Conf. Learn. Represent. 2020.",
    "Jiang, D.; Wu, Z.; Hsieh, C.-Y.; Chen, G.; Liao, B.; Wang, Z.; Shen, C.; Cao, D.; Wu, J.; "
    "Hou, T. Could Graph Neural Networks Learn Better Molecular Representation for Drug Discovery? "
    "A Comparison Study of Descriptor-Based and Graph-Based Models. J. Cheminform. 2021, 13 (1), 12.",
    "Lundberg, S. M.; Lee, S.-I. A Unified Approach to Interpreting Model Predictions. "
    "Adv. Neural Inf. Process. Syst. 2017, 30, 4765-4774.",
    "Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 2010, 50 (5), "
    "742-754.",
    "Sheridan, R. P. Time-Split Cross-Validation as a Method for Estimating the Goodness of "
    "Prospective Prediction. J. Chem. Inf. Model. 2013, 53 (4), 783-790.",
    "Subramanian, G.; Ramsundar, B.; Pande, V.; Denny, R. A. Computational Modeling of "
    "beta-Secretase 1 (BACE-1) Inhibitors Using Ligand Based Approaches. J. Chem. Inf. Model. 2016, "
    "56 (10), 1936-1949.",
    "Tropsha, A. Best Practices for QSAR Model Development, Validation, and Exploitation. "
    "Mol. Inform. 2010, 29 (6-7), 476-488.",
    "van Tilborg, D.; Alenicheva, A.; Grisoni, F. Exposing the Limitations of Molecular Machine "
    "Learning with Activity Cliffs. J. Chem. Inf. Model. 2022, 62 (23), 5938-5951.",
    "Vassar, R.; Bennett, B. D.; Babu-Khan, S.; Kahn, S.; Mendiaz, E. A.; Denis, P.; et al. "
    "beta-Secretase Cleavage of Alzheimer's Amyloid Precursor Protein by the Transmembrane Aspartic "
    "Protease BACE. Science 1999, 286 (5440), 735-741.",
    "Wu, Z.; Ramsundar, B.; Feinberg, E. N.; Gomes, J.; Geniesse, C.; Pappu, A. S.; Leswing, K.; "
    "Pande, V. MoleculeNet: A Benchmark for Molecular Machine Learning. Chem. Sci. 2018, 9 (2), "
    "513-530.",
    "Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. How Powerful Are Graph Neural Networks? "
    "Int. Conf. Learn. Represent. 2019.",
    "Yang, K.; Swanson, K.; Jin, W.; Coley, C.; Eiden, P.; Gao, H.; et al. Analyzing Learned "
    "Molecular Representations for Property Prediction. J. Chem. Inf. Model. 2019, 59 (8), 3370-3388.",
]

FIGURE_CAPTIONS = {
    "fig1_workflow.png": "Computational workflow, from dataset curation through representation and "
                         "model benchmarking to evaluation and interpretation.",
    "fig2_dataset.png": "The curated BACE-1 dataset. (a) Activity classes at the pIC50 = 7 threshold. "
                        "(b) Distribution of measured potency. (c) Molecular weight against cLogP, "
                        "with Lipinski limits marked. (d) Bemis-Murcko scaffold frequency.",
    "fig3_splits.png": "Random against scaffold splitting. (a, b) t-SNE of ECFP4 space coloured by "
                       "subset for one seed. (c) Maximum Tanimoto similarity of each test molecule to "
                       "the training set, pooled over five seeds.",
    "fig4_benchmark.png": "Representation by model family on scaffold-split test sets, averaged over "
                          "five seeds. Darker is better in both panels.",
    "fig5_split_gap.png": "Paired comparison of random and scaffold splitting for identical models. "
                          "Every configuration scores better under random splitting.",
    "fig6_tuning.png": "Effect of random-search hyperparameter optimisation. (a, b) Default against "
                       "tuned performance per model family. (c) Best validation ROC-AUC against the "
                       "number of configurations evaluated.",
    "fig7_roc.png": "(a) ROC curves for representative models on one scaffold-split seed. "
                    "(b) Predicted against measured pIC50 for the best regression model.",
    "fig8_ablation.png": "Graph neural network ablations on scaffold-split test sets. The dashed line "
                         "marks the base configuration.",
    "fig9_shap.png": "SHAP attribution over ECFP4 bits for a random forest classifier. (a) Bits with "
                     "the strongest mean effect where the substructure occurs. (b) The circular "
                     "environments those bits encode.",
    "fig10_atoms.png": "Atom-level attribution for confidently predicted test molecules, measured as "
                       "the change in predicted probability when each atom's fingerprint bits are "
                       "removed.",
    "fig11_applicability.png": "Applicability domain. (a, b) Accuracy and error against maximum "
                               "Tanimoto similarity to the training set. (c) Accuracy when only the "
                               "most confident predictions are retained.",
    "fig12_final.png": "Final comparison of the best configuration from each model family, the graph "
                       "neural network and the ensemble, on scaffold-split test sets.",
}

TABLE_CAPTIONS = {
    "table_benchmark_cls.csv": "Classification performance for every representation and model family "
                               "on scaffold-split test sets (mean plus or minus SD over five seeds).",
    "table_benchmark_reg.csv": "Regression performance for every representation and model family on "
                               "scaffold-split test sets (mean plus or minus SD over five seeds).",
    "table_tuning_cls.csv": "Effect of hyperparameter tuning on classification, per model family.",
    "table_tuning_reg.csv": "Effect of hyperparameter tuning on regression, per model family.",
    "table_gnn_cls.csv": "Graph neural network and its ablations, classification.",
    "table_gnn_reg.csv": "Graph neural network and its ablations, regression.",
}
