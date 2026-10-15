# Public Omics Datasets on Trained Immunity and Innate Immune Activation States

Scope note: every accession below was verified live against NCBI E-utilities (GEO `gds` esummary / PubMed elink), the EBI PRIDE v3 API, the Metabolomics Workbench REST API, the EBI BioStudies API, EGA, or the authors' own data portal. Sample counts (`n=`), species and assay-type strings are taken verbatim from the repository records. Where I could not verify an accession I say so explicitly in **Gaps** rather than guessing. Stimulation-protocol detail is given only where the repository record or paper text supplied it; GEO summaries frequently omit concentrations, so protocol fields are often incomplete and flagged.

---

## Key Question 1 — The Radboud/Netea canon: enumerate accessions from the major trained-immunity papers

### Takeaway
Essentially the entire Netea/Stunnenberg/Logie "trained immunity canon" is deposited **open-access in GEO**, and it is unusually well suited to multi-omic modelling because most papers deposited RNA-seq + histone ChIP-seq (+ ATAC-seq, WGBS, or methylation arrays) as SuperSeries generated on the *same* primary human monocyte donors. The two exceptions are the 300BCG raw ATAC-seq (controlled, EGA) and the 500FG genotypes (controlled, EGA); metabolomics/Seahorse/cytokine data from these papers are largely *not* in dedicated repositories.

### Cited Findings

**Saeed et al., Science 2014 — "Epigenetic programming of monocyte-to-macrophage differentiation and trained innate immunity" (PMID 25258085)**
- GEO **GSE58310**, `n=64`, *Homo sapiens*, "Expression profiling by high throughput sequencing; Genome binding/occupancy profiling by high throughput sequencing" (i.e. RNA-seq + histone ChIP-seq), released 2014/09/26. Series summary: comprehensive epigenetic profiling of naïve monocytes, monocyte-derived macrophages, β-glucan-trained and LPS-tolerized states; "β-Glucan trains innate immune cells through extensive remodeling of distal regulatory region-bound histone acetylation, resulting in a sizeable exclusive epigenomic signature"; includes genome-wide TF footprint analysis. Open access. — [NCBI eutils esummary, GEO GSE58310](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58310); [PubMed 25258085](https://pubmed.ncbi.nlm.nih.gov/25258085/)

**Novakovic et al., Cell 2016 — "β-Glucan Reverses the Epigenetic State of LPS-Induced Immunological Tolerance" (PMID 27863248)**
- SuperSeries GEO **GSE85246**, `n=239`, *Homo sapiens*, "Epigenome maps of time-resolved monocyte to macrophage differentiation and innate immune memory". Series summary states explicitly: "**ChIP-seq, RNA-seq, WGBS and ATAC-seq data were generated**" to characterise LPS-induced tolerance *in a time-dependent manner*, identifying "epigenetic programs in tolerance and trained macrophages, and the potential transcription factors involved."
- Subseries: **GSE85243** (RNA-seq, `n=94`), **GSE85245** (ChIP-seq, `n=132`), **GSE87218** (ATAC-seq, `n=13`). All human, all released 2016/11/18, all open. — [GEO GSE85246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85246); [PubMed 27863248](https://pubmed.ncbi.nlm.nih.gov/27863248/)

**Cheng et al., Science 2014 — "mTOR- and HIF-1α-mediated aerobic glycolysis as metabolic basis for trained immunity" (PMID 25258083)**
- Only one GEO series is linked: **GSE57206**, `n=12`, ***Mus musculus***, RNA-seq, "The role of HIF-1 in beta-glucan induced response in myeloid cell" (ArrayExpress mirror **E-GEOD-57206**). The human Seahorse/metabolomics data underpinning this paper are not in a repository I could locate. — [PubMed elink 25258083 → GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57206); [EBI BioStudies E-GEOD-57206](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-GEOD-57206)

**Quintin et al., Cell Host Microbe 2012 — "Candida albicans infection affords protection against reinfection via functional reprogramming of monocytes" (PMID 22901542)**
- GEO **GSE34260**, `n=15`, ***Mus musculus; Homo sapiens*** (mixed), RNA-seq + ChIP-seq, "Candida albicans induces monocyte training via epigenetic programming through a dectin 1-dependent pathway", released 2012/12/21. Open. — [GEO GSE34260](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE34260)

**Arts et al., Cell Metabolism 2016 — "Glutaminolysis and Fumarate Accumulation Integrate Immunometabolic and Epigenetic Programs in Trained Immunity" (PMID 27866838)**
- GEO **GSE86940**, `n=9`, *Homo sapiens*, RNA-seq + ChIP-seq: "Histone modification and RNA changes in macrophages exposed to Beta-glucan and Fumarate". Open. — [PubMed elink 27866838 → GEO GSE86940](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE86940)

**Arts et al., Cell Reports 2016 — "Immunometabolic Pathways in BCG-Induced Trained Immunity" (PMID 27926861)**
- **No GEO series is linked to this PMID** (verified by PubMed→GEO elink, which returned no hits). This paper's data are primarily metabolomics/Seahorse/cytokine and appear not to be deposited. — [PubMed 27926861](https://pubmed.ncbi.nlm.nih.gov/27926861/)

**Bekkering et al., Cell 2018 — "Metabolic Induction of Trained Immunity through the Mevalonate Pathway" (PMID 29328908)**
- GEO **GSE104227**, `n=12`, *Homo sapiens*, RNA-seq + ChIP-seq: "Epigenomes and Transcriptomes of human macrophages after ex vivo exposure to mevalonate" — healthy-donor monocytes exposed ex vivo to mevalonate, gene expression and histone-modification changes.
- GEO **GSE104282**, `n=24`, *Homo sapiens*, RNA-seq + ChIP-seq: "Epigenomes and Transcriptomes of human monocytes from Hyper IgD Syndrome (HIDS) patients" (the genetic-mevalonate-kinase-deficiency arm). Both open. — [GEO GSE104227](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104227); [GEO GSE104282](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104282)

**Arts et al., Cell Host Microbe 2018 — BCG vaccination, epigenetic reprogramming of monocytes and protection against yellow-fever-vaccine viraemia (PMID 29324233)**
- GEO **GSE104149**, `n=26`, *Homo sapiens*, RNA-seq + ChIP-seq: "Epigenomes and transcriptomes of human monocytes **before and after in vivo exposure** to Bacillus Calmette-Guérin vaccine"; summary: "we apply epigenomic and transcriptomic analysis to a clinical trial of BCG vaccination in healthy adults." The record is linked to two PMIDs (29324233 and 31671203), i.e. it is reused across two Netea-group papers. Open. — [GEO GSE104149](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104149)

**Christ et al., Cell 2018 — "Western Diet Triggers NLRP3-Dependent Innate Immune Reprogramming" (PMID 29328911) — mouse**
- SuperSeries **GSE97926**, `n=48`, *Mus musculus*, RNA-seq + ChIP/ATAC; subseries **GSE97925** (RNA-seq, `n=24`) and **GSE97924** (ATAC-seq, `n=5`); plus **GSE106789** (RNA-seq II, `n=19`). All mouse, all open. — [PubMed elink 29328911 → GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE97926)

**Fanucchi et al., Nature Genetics 2019 — "Immune genes are primed for robust transcription by proximal long noncoding RNAs located in nuclear compartments" (UMLILO; Mhlanga lab; PMID 30531872; publisher correction PMID 30647470)**
- Only **GSE120621** is linked in GEO: `n=2`, *Homo sapiens*, RNA-seq, "Transcriptome measurements after knocking out the UMLILO lncRNA" (released 2018/10/15). A GEO author search for `Mhlanga[Author]` returns no 4C-seq/Hi-C/ChIP-seq series from 2018–2019 for this paper. — [GEO GSE120621](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120621); [PubMed 30531872](https://pubmed.ncbi.nlm.nih.gov/30531872/)

**Cirovic et al., Cell Host Microbe 2020 — "BCG Vaccination in Humans Elicits Trained Immunity via the Hematopoietic Progenitor Compartment" (PMID 32544459)**
- SuperSeries **GSE124220**, `n=75`, *Homo sapiens*, RNA-seq + ATAC-seq. Subseries: **GSE124217** (PBMC RNA-seq, `n=10`), **GSE124218** (bone-marrow RNA-seq, `n=39`), **GSE124219** (ATAC-seq, `n=26`). Human bone-marrow aspirates plus peripheral blood from BCG-vaccinated volunteers. Open. — [GEO GSE124220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124220)

**Moorlag et al., Cell Reports 2020 — "BCG Vaccination Induces Long-Term Functional Reprogramming of Human Neutrophils" (PMID 33207187)**
- SuperSeries **GSE153729**, `n=16`, human; subseries **GSE153727** (neutrophil RNA-seq, `n=6`) and **GSE153728** (neutrophil **H3K4me3 ChIP-seq**, `n=10`), "before and after in vivo exposure to Bacillus Calmette-Guérin vaccine". Open. — [GEO GSE153729](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153729)

**Moorlag et al., Cell Reports 2019 — "β-Glucan-Induced Trained Immunity Protects against Leishmania braziliensis Infection: a Crucial Role for IL-32" (PMID 31484076)**
- GEO **GSE132155**, `n=40`, *Homo sapiens*, RNA-seq. Open. — [GEO GSE132155](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132155)

**Domínguez-Andrés / itaconate–fumarate axis (Netea group)**
- GEO **GSE212282**, `n=75`, *Homo sapiens*, RNA-seq (PMID 37330914): "Transcriptional remodeling of primary human monocytes exposed to itaconate and its derivatives in the context of trained immunity"; summary states the design profiled "human monocytes and in vitro differentiated macrophages, exposed to a combination of **LPS, BG [β-glucan], DMI, or Itaconate**." Open. — [GEO GSE212282](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212282)
- GEO **GSE185433**, `n=39`, *Homo sapiens*, RNA-seq (PMID 37291433): "Transcriptional remodeling of primary human monocytes trained with interleukin-4 (IL4)" — macrophages trained with IL-4 then re-exposed to LPS (a non-microbial training arm directly relevant to M2-type training). Open. — [GEO GSE185433](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185433)

**Adjacent canon papers (mouse) worth having for cross-species signature transfer**
- Kaufmann et al., Cell 2018, "BCG Educates Hematopoietic Stem Cells…" (PMID 29328912): SuperSeries **GSE98600** (`n=85`, mouse) = **GSE98599** (RNA-seq, `n=32`), **GSE98598** (ChIP-seq, `n=44`), **GSE98597** (ATAC-seq, `n=7`); plus **GSE108067** (scRNA-seq, `n=2`). — [GEO GSE98600](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98600)
- Mitroulis et al., Cell 2018, "Modulation of Myelopoiesis Progenitors Is an Integral Component of Trained Immunity" (PMID 29328910): **GSE95617**, `n=31`, mouse, RNA-seq of HSPCs after β-glucan administration. — [GEO GSE95617](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE95617)
- Kalafati et al., Cell 2020 (PMID 33125892): **GSE152353** (**scATAC-seq**, `n=4`, mouse GMP + neutrophils after β-glucan), **GSE139456** (GMP RNA-seq, `n=7`), **GSE139451** (tumour-associated monocyte RNA-seq, `n=9`). — [GEO GSE152353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152353)

**Netea-group time-resolved BCG multi-modal block (critical for multi-omic modelling; PMID 35930640 / open)**
- **GSE168468** — "Time-resolved transcriptomes of BCG vaccine induced trained immunity in human monocytes", `n=85`, human RNA-seq; "profiled the transcriptomes of human monocytes responding to **two different BCG vaccine strains**".
- **GSE168628** — "DNA methylation dynamics during monocyte-to-macrophage differentiation and BCG vaccine induced trained immunity", `n=45`, methylation array.
- **GSE168779** — "Epigenome-Wide Association Study of Bacillus Calmette Guerin vaccination in human circulating monocytes", `n=130`, methylation (genome tiling array).
- **GSE183528** — "Time-resolved post-translational histone modifications during BCG vaccine induced trained immunity in human monocytes", `n=37`, ChIP-seq.
- **GSE166238** — "Transcriptomes of human monocytes exposed to **oxLDL** ex vivo", `n=21`, RNA-seq (PMIDs 34267957, 35930640).
- **GSE126352** — "Transcriptomes of human monocytes from patients with familial hypercholesterolemia before and after statin treatment", `n=15`.
— [GEO GSE168468](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE168468); [GEO GSE166238](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE166238)

**Histone-lactylation layer (Netea/Novakovic/Mhlanga, 2025, open)**
- **GSE262096** — "Remodeling of Histone lactylation, acetylation, and methylation in circulating monocytes **by BCG vaccination**", `n=33`, human ChIP-type.
- **GSE262097** — "Remodeling of Histone 3 Lysine 18 lactylation (H3K18lac) in monocytes-to-macrophage differentiation and induction of Trained Immunity by BCG **in vitro**", `n=37`.
- **GSE262098** — "Transcriptomes during monocyte-to-macrophage differentiation and BCG induced trained immunity in vitro", `n=21`, RNA-seq.
- **GSE262099** — "Effect of Lactate dehydrogenase inhibitor (LDHi; GSK2837808A) on BCG vaccine induced trained immunity in vitro", `n=28`, RNA-seq.
— [GEO GSE262096](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262096)

### Inferences
- GSE58310 (Saeed) and the GSE85246 family (Novakovic) are the two highest-value anchors for a β-glucan-training signature: both pair transcriptome with the exact histone marks requested (H3K4me3/H3K27ac/H3K4me1 appear in the ChIP-seq subseries), on the same donors, with LPS-tolerance arms in the same experiment, so a trained-vs-tolerant contrast can be computed within-donor.
- The 2021–2025 Netea/Novakovic GEO depositions (GSE166238, GSE168468/168628/168779/183528, GSE262096–99) effectively form a single growing human resource built on the same protocol family (monocyte → 24 h training stimulus → 5–6 d rest → restimulation), which makes them the most internally consistent set for building a target signature.

### Gaps
- **Fanucchi et al. 2019 (UMLILO) chromatin data**: I could verify only GSE120621 (`n=2`). The paper's 4C-seq, Hi-C/chromatin-conformation and H3K4me3 ChIP-seq accessions could not be verified — the Nature Genetics article page is paywalled/redirects to an auth endpoint, the PMC record (PMC6353842) returned no retrievable data-availability text, and web search returned no matching accession. Do not cite a specific 4C/Hi-C accession for this paper without checking the article's "Accession codes" section directly.
- **Arts et al. Cell Reports 2016 (Immunometabolic Pathways in BCG-Induced Trained Immunity)**: no GEO deposit exists per PubMed→GEO elink; its metabolomics is not in Metabolomics Workbench or MetaboLights under searchable titles (see Q6).
- **Exact stimulation parameters** (β-glucan µg/mL, BCG MOI, 24 h priming vs washout, 5–6 day rest, LPS restimulation dose/duration) are *not* machine-readable in most GEO records. They must be read from each paper's STAR Methods; I could not retrieve the Netea STAR Protocols in-vitro-training protocol text (PMC7921712) within this session.
- **Seahorse / extracellular-flux and cytokine-ELISA raw data** for the canon papers: I found no repository deposits. These are published as figures/supplementary tables only.

---

## Key Question 2 — Human cohort resources: 300BCG, 500FG / HFGP, BCG-CORONA / ACTIVATE, BCG revaccination. What is deposited, where, open or controlled?

### Takeaway
300BCG is **split-access**: processed ATAC-seq counts are open in GEO (GSE241092), raw ATAC-seq is controlled in EGA (EGAS50000000090), and immune phenotyping is open on Zenodo. 500FG genotypes are controlled in EGA behind an HFGP data-access committee. I found **no deposited omics for the BCG-CORONA or ACTIVATE COVID-19 BCG trials**.

### Cited Findings
- **300BCG flagship paper**: Moorlag SJCFM, Folkman L, ter Horst R, Krausgruber T, et al., "Multi-omics analysis of innate and adaptive responses to BCG vaccination reveals epigenetic cell states that predict trained immunity", *Immunity* 2024;57(1), doi:10.1016/j.immuni.2023.12.005, **PMID 38198850**; senior authors Mihai G. Netea and Christoph Bock (Bock = lead contact). — [300BCG project portal, medical-epigenomics.org](https://medical-epigenomics.org/papers/300bcg/)
- **300BCG deposits** (from the authors' own portal): GEO **GSE241092** = *processed ATAC-seq count data*; EGA **EGAS50000000090** = *raw ATAC-seq data*; Zenodo **DOI 10.5281/zenodo.10288920** = *immunological phenotyping data*; plus a UCSC GRCh38 track hub for all samples and GitHub repos `epigen/300BCG_ATACseq_pipeline` and `epigen/300BCG_analysis`. — [300BCG portal](https://medical-epigenomics.org/papers/300bcg/)
- **GSE241092 record**: `n=848` samples, *Homo sapiens*, "Genome binding/occupancy profiling by high throughput sequencing", released 2024/01/08. Summary: "we immunized healthy individuals with Bacillus Calmette-Guérin (BCG) … We established personal immune profiles and chromatin accessibility maps over a time course of BCG vaccination in **323 individuals** … BCG vaccination enhances the innate immune response **only in individuals with dormant immune states at baseline**". — [GEO GSE241092](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241092)
- **300BCG assays and design**: ATAC-seq (chromatin accessibility), single-cell RNA-seq (with and without LPS stimulation), cytokine production capacity, circulating inflammatory markers, immune cell frequencies; **323 healthy individuals**, 90-day time course with **baseline (day 0), day 14 and day 90** comparisons. The portal does not mention metabolite profiling and does not name flow cytometry as a method. — [300BCG portal](https://medical-epigenomics.org/papers/300bcg/)
- **500FG / Human Functional Genomics Project genotypes**: EGA dataset **EGAD00010002149**, Data Access Committee **EGAC00001002135**; genotype data from healthy Dutch individuals on Illumina HumanOmniExpress Exome-8v1.0 SNP chip, **482 samples**, record dated 18/06/2021. Access requires a data-request form emailed to the HFGP contact; a committee (Netea, Joosten, Wijmenga, Xavier) meets monthly, and a signed Data Access Agreement is required before transfer. **Controlled access.** — [EGA EGAD00010002149](https://ega-archive.org/datasets/EGAD00010002149)
- **BCG-vaccinated human bone-marrow progenitors, single-cell (open)**: SuperSeries **GSE248730** (`n=42`, human, PMID 39153479) "BCG vaccination alters the epigenetic landscape of progenitor cells in human bone marrow to influence innate immune responses", with subseries **GSE248728** (**scRNA-seq**, `n=28`) and **GSE248720** (**scATAC-seq**, `n=14`). — [GEO GSE248730](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248730)
- **BCG in bladder cancer (TRIBUTE)**: SuperSeries **GSE190530** (`n=55`, human) "TRained Immunity induced by Bcg in UroThElial carcinoma (Tribute)" = **GSE190529** (RNA-seq, `n=21`) + **GSE190528** (epigenetic remodeling / ChIP-type, `n=34`). Open; no PMID linked in the GEO record. — [GEO GSE190530](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190530)
- **BCG-related DNA methylation cohort**: **GSE182215**, `n=38`, human, methylation (genome tiling array): "DNA methylation profiling of immune cells from tuberculosis-exposed individuals overlaps with BCG-induced epigenetic changes…" (released 2024/08/02). — [GEO GSE182215](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE182215)
- **BCG long-term effects**: **GSE300272**, `n=20`, human RNA-seq, "LONG-TERM EFFECTS OF BCG VACCINATION ON TELOMERE LENGTH AND TELOMERASE ACTIVITY" (2025/07/28). — [GEO GSE300272](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE300272)
- **ImmPort**: a targeted search surfaced only **SDY2942** (DR59 release, December 2025), which is *intravenous BCG in rhesus macaques* and lung NK cells — not human trained immunity. No ImmPort SDY could be matched to Moorlag 2024 or Arts 2018. — [ImmPort DR59 release notes](https://docs.immport.org/data/release/notes/DR59_DataRelease/)

### Inferences
- For 300BCG, a modelling pipeline that needs only chromatin-accessibility matrices plus immune phenotypes can stay entirely open-access (GSE241092 + Zenodo 10.5281/zenodo.10288920); only re-processing from raw reads requires an EGA application (EGAS50000000090).
- The absence of HFGP cytokine/metabolite QTL raw data in a public repository (only genotypes are in EGA) means 500FG-derived cytokine-response phenotypes will likely have to be taken from paper supplementary tables rather than a repository.

### Gaps
- **BCG-CORONA and ACTIVATE trials**: no GEO series matched queries for BCG/COVID-19 healthcare-worker or elderly-BCG randomized trials, and I found no EGA/ImmPort record for either. Either the omics were not deposited or they are deposited under titles that do not mention the trial names. Flag as unverified/not found.
- **BCG revaccination studies**: no specific deposit identified. GSE104149 and the GSE168468 family cover primary BCG vaccination, not revaccination.
- **Flow/CyTOF**: GEO queries for "CyTOF mass cytometry BCG vaccination trained immunity" returned nothing, and a FlowRepository-targeted search found no FR-FCM accession for β-glucan/BCG monocyte training. I found **no verified FlowRepository or ImmPort accession for any human trained-immunity flow/CyTOF dataset**. (Note: FlowRepository accessions of the form FR-FCM-xxxx do exist for other immunology papers — e.g. FR-FCM-ZYRD for an unrelated fetal-intestine CyTOF study — so the repository is in use, just not findable for this topic via these searches.)
- **Seahorse/extracellular-flux data for human cohorts**: none found in any repository.

---

## Key Question 3 — Macrophage polarization reference atlases

### Takeaway
Xue et al. Immunity 2014 is fully open as a four-part GEO SuperSeries (GSE47189 / GSE46903 / GSE47188 / GSE51307) and remains the canonical human macrophage activation spectrum (28 stimuli incl. LPS, IFN-γ, IL-4). It can be complemented with GSE100382 (IFN/TNF chromatin reprogramming), Monaco GSE107011 (29 sorted immune cell types) and DICE GSE118165 (resting + stimulated ATAC-seq/RNA-seq).

### Cited Findings
- **Xue et al., Immunity 2014, "Transcriptome-based network analysis reveals a spectrum model of human macrophage activation" (PMID 24530056; Schultze lab)**: SuperSeries **GSE47189**, `n=412`, *Homo sapiens*, "Expression profiling by array; Genome binding/occupancy profiling by high throughput sequencing", released 2014/02/13. Subseries: **GSE46903** (expression arrays, `n=384`, also flagged "Third-party reanalysis"), **GSE47188** (**ChIP-seq**, `n=8`), **GSE51307** (**miRNA-seq**, `n=20`). All open. — [GEO GSE47189](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE47189)
- **GSE100382**, `n=24`, *Homo sapiens*, RNA-seq (PMID 28825701): "Type I IFNs and TNF Cooperatively Reprogram Epigenomic Landscape of Human Macrophages to Promote Inflammatory Activation". Summary: "A comprehensive approach integrating **RNA-seq, ChIP-seq and ATAC-seq digital footprinting** showed that TNF and type I IFNs extensively remodel chromatin states in human macrophages". — [GEO GSE100382](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100382)
- **Monaco et al. (PMIDs 30528453, 30726743)**: **GSE107011**, `n=127`, *Homo sapiens*, "RNA-Seq profiling of 29 immune cell types and peripheral blood mononuclear cells"; 29 sorted PBMC types from 4 Singaporean-Chinese individuals (S4 cohort) plus RNA-seq and microarray on PBMCs from an extended 13-individual cohort (S13). — [GEO GSE107011](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107011)
- **DICE (PMID 31570894)**: **GSE118165**, `n=166`, *Homo sapiens*; summary: "we collected **ATAC-seq and RNA-seq** data from immune cell types under **resting and stimulated** conditions from blood of healthy [donors]" — explicitly addressing the gap that epigenome mapping had focused on resting ex vivo immune cells. — [GEO GSE118165](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118165)
- **ImmGen**: ArrayExpress/BioStudies **E-GEOD-15907**, "Immunological Genome Project data Phase 1" (mouse). — [EBI BioStudies E-GEOD-15907](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-GEOD-15907)
- **Macrophage polarization × endotoxin tolerance crossover**: **GSE189847**, `n=28`, *Homo sapiens*, RNA-seq (PMID 36240188): "A transcriptomic analysis of the effects of **macrophage polarization and endotoxin tolerance** on the response to Salmonella"; summary contrasts M1 (non-permissive) vs M2 (permissive) states with endotoxin tolerance. — [GEO GSE189847](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189847)
- **Glucocorticoid-polarized human monocyte/macrophage epigenomes (Logie/Stunnenberg, same protocol family as the trained-immunity series)**: **GSE109440** (`n=72`, ChIP-type, "Extensive epigenomic integration of the glucocorticoid response in primary human monocytes and in vitro derived macrophages"), **GSE109439** (RNA-seq, `n=16`), **GSE109438** (`n=32`, epigenome + GR binding). — [GEO GSE109440](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109440)
- **Human macrophage/monocyte cell-source reference (Novakovic, 2026)**: **GSE261696** (RNA-seq, `n=19`) and **GSE261697** (ATAC-seq, `n=17`) — "Transcriptional/Chromatin accessibility characteristics of human primary, bone marrow-derived and iPSC-derived monocytes and macrophages". Useful for checking whether a signature transfers across macrophage model systems. — [GEO GSE261696](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261696)

### Inferences
- GSE46903 (384 arrays) gives the densest single-platform stimulus grid for anchoring an M1/M2/"spectrum" axis; GSE118165 (DICE) is the better choice when the signature must be chromatin-level and stimulation-aware.

### Gaps
- **Martinez/Gordon polarization sets**: I did not verify a specific accession for these and will not guess one.
- **BLUEPRINT monocyte/macrophage epigenomes**: GEO text search for "BLUEPRINT monocyte macrophage epigenome" returned nothing; BLUEPRINT data are distributed via the BLUEPRINT Data Analysis Portal and EGA rather than GEO, and I could not verify a specific EGAS/EGAD accession in this session. Note, however, that GSE58310 (Saeed 2014) was produced within the BLUEPRINT consortium framework and *is* open in GEO.
- **ENCODE monocyte data**: not verified here; ENCODE uses ENCSR/ENCFF identifiers that I did not query. Do not cite specific ENCSR accessions from these notes.
- **Human Cell Atlas monocyte/macrophage**: Mulder K. et al., "Cross-tissue single-cell landscape of human monocytes and macrophages in health and disease", *Immunity* 2021;54(8):1883-1900.e5, **PMID 34331874** (MNP-VERSE compendium, ~178,651 mononuclear phagocytes from 13 tissues across 41 datasets) — **no GEO/ArrayExpress/EGA accession could be verified** for the integrated object. — [UWA research repository record](https://research-repository.uwa.edu.au/en/publications/cross-tissue-single-cell-landscape-of-human-monocytes-and-macroph/)
- **Important disambiguation for the report-writer**: the brief refers to "Mulder et al. Cell 2021 (300BCG / human functional genomics)". I could not find such a paper. The 300BCG flagship multi-omics paper is **Moorlag et al., Immunity 2024 (PMID 38198850)**; the Mulder 2021 paper is the *Immunity* cross-tissue monocyte/macrophage atlas (PMID 34331874), which is unrelated to 300BCG. Treating them as one paper would be an error.

---

## Key Question 4 — Endotoxin tolerance datasets (human monocytes, in vivo human endotoxemia, sepsis immunoparalysis)

### Takeaway
There is a deep open-access stack: in vitro human monocyte LPS tolerance with matched multi-omics (GSE172116, GSE85246 family, GSE327840), the Calvano/Lowry "Inflammation and Host Response to Injury" in vivo endotoxemia and trauma arrays (GSE3284, GSE36809, GSE11375), and sepsis immunoparalysis cohorts (GSE205672, GSE46914, GSE63311).

### Cited Findings
**In vitro / ex vivo human LPS tolerance**
- **GSE172116**, `n=99`, *Homo sapiens*, "Expression profiling by high throughput sequencing; Genome binding/occupancy profiling…" (PMID 35751107; Knight lab): "we systematically delineate the acute innate immune response to endotoxin in terms of human macrophage enhancer activity and contrast with **endotoxin tolerance**, profiling the coding and non-coding transcriptome, **chromatin accessibility** and **epigenetic modifications**"; includes eRNA signatures and context-specific eQTL. Open. — [GEO GSE172116](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE172116)
- **GSE327840**, `n=58`, *Homo sapiens*, RNA-seq, released 2026/04/18, **no PMID yet**: "RNA-seq profiling of LPS tolerance in monocytes, THP-1, fibroblasts, smooth muscle cells, and endothelial cells"; design = "single and double LPS stimulation"; findings: primary monocytes and smooth muscle cells developed tolerance, THP-1 limited (TNF only), endothelial cells none; "No conserved transcriptional tolerance signature was observed across tolerant cell types; overlap was limited to three genes (**OAS1, IFIT2, IFIT3**)." Open. — [GEO GSE327840](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE327840)
- **GSE22248**, `n=12`, human, expression array: "Genome-wide gene expression analysis during endotoxin tolerance in PBMCs" (2011). ArrayExpress mirror **E-GEOD-22248**. — [GEO GSE22248](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22248)
- **GSE159249** (RNA-seq, `n=12`) and **GSE159227** (methylation tiling array, `n=12`): "The JAK2-STAT pathway epigenetically regulates tolerized genes during the first encounter to bacterial antigens" — matched transcriptome + methylome. — [GEO GSE159249](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159249)
- **GSE277003**, `n=68`, *Homo sapiens*, methylation (genome tiling array), PMID 39348545: "TCA metabolism regulates DNA hypermethylation in LPS and *Mycobacterium tuberculosis*–induced immune tolerance"; includes data from TB patients receiving everolimus plus standard antibiotics. Directly links metabolism to the methylome in tolerance. — [GEO GSE277003](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277003)
- **GSE240674**, `n=36`, *Homo sapiens*, "Expression profiling by high throughput sequencing; Other": "Non-canonical function of histone methyltransferase **G9a** in the translational regulation of chronic inflammation". — [GEO GSE240674](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240674)
- **GSE108239**, `n=45`, human expression array: "Analysis of HERV transcriptome in PBMCs upon **dose dependent LPS** stimulations". — [GEO GSE108239](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108239)
- **GSE135753**, `n=12`, human RNA-seq: "NAPRT activation of macrophages". — [GEO GSE135753](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135753)
- **GSE124928**, `n=48`, human RNA-seq: "Transcriptomic profiling of osteoarthritis synovial macrophages reveals a **tolerized phenotype** compounded by a weak corticosteroid response" (Logie-group linked). — [GEO GSE124928](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124928)

**In vivo human endotoxemia / trauma (Inflammation and Host Response to Injury "Glue Grant", Calvano/Lowry/Tompkins)**
- **GSE3284**, `n=110`, *Homo sapiens*, expression array, PMIDs **16136080** (Calvano et al., "A network-based analysis of systemic inflammation in humans", Nature 2005), 20233835, 19274080, 23383223. Title in GEO: "A network-based analysis of systemic inflammation in humans". Open. — [GEO GSE3284](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE3284)
- **GSE35590**, `n=16`, human expression array: "Transcriptome kinetics of circulating **neutrophils** during human experimental endotoxemia" (2012). — [GEO GSE35590](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35590)
- **GSE36809**, `n=857`, human expression array, "A genomic storm in critically injured humans" (Glue Grant trauma cohort, 2012); and **GSE11375**, `n=184`, "A Genomic Score Prognostic of Outcome in Trauma Patients". Both open. — [GEO GSE36809](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36809)

**Sepsis immunoparalysis**
- **GSE205672**, `n=460`, *Homo sapiens*, RNA-seq, PMID 39566468: "Transcriptomes of human PBMCs and monocytes in sepsis patients" — summary names the **FUSE cohort** (Netea-group linked). Open. — [GEO GSE205672](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205672)
- **GSE46914**, `n=24`, human expression array, PMID 23874546: "Identification of biomarkers of response to IFNG during endotoxin tolerance: application to septic shock" — i.e. IFN-γ reversal of tolerance, directly relevant to immunoparalysis reversal signatures. — [GEO GSE46914](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE46914)
- **GSE63311** / ArrayExpress **E-GEOD-63311**: "An Endotoxin Tolerance Signature Predicts Sepsis and Organ Dysfunction at Initial Clinical Presentation". — [EBI BioStudies E-GEOD-63311](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-GEOD-63311)
- **GSE212093** (SuperSeries, `n=48`, human) "Systemic Inflammation Impairs Human Myelopoiesis" = **GSE212092** (scRNA-seq, `n=18`) + **GSE212091** (bulk RNA-seq, `n=30`); Netea/Stunnenberg linked. — [GEO GSE212093](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212093)

### Inferences
- GSE172116 is the single best "tolerance" counterpart to GSE58310/GSE85246 for a trained-vs-tolerized contrast because it carries transcriptome + accessibility + histone modifications in the same human macrophage system with an explicit acute-vs-tolerant design.
- GSE327840's negative result (only OAS1/IFIT2/IFIT3 shared across tolerant cell types) is a useful caution: an LPS-tolerance signature derived in monocytes should not be assumed portable to stromal/endothelial cells.

### Gaps
- The classic Calvano human endotoxemia *in vivo* LPS-challenge arm and the trauma arm are bundled across GSE3284/GSE36809/GSE11375 with four associated PMIDs; I did not resolve which exact GSM subsets correspond to the 4 ng/kg LPS challenge vs. trauma patients. That mapping needs a sample-level check.
- No proteomics or metabolomics deposit was found for human endotoxemia within this session.

---

## Key Question 5 — Single-cell atlases of monocyte states after BCG or β-glucan

### Takeaway
Three human single-cell resources are open and directly on-topic: GSE184241 (scRNA-seq of monocytes before/after BCG with LPS restimulation), GSE248728/GSE248720 (scRNA-seq + scATAC-seq of BCG-vaccinated human bone-marrow progenitors), and the 300BCG scRNA-seq component. For β-glucan, single-cell data are currently mouse (GSE152353 scATAC).

### Cited Findings
- **GSE184241**, `n=18`, *Homo sapiens*, RNA-seq (single-cell), PMID 34788625: "Single-cell transcriptomic profiles reveal changes associated with BCG-induced trained immunity and protective effects in circulating monocytes". Design: "single-cell transcriptomic measurements **before and after BCG vaccination** using secondary immune stimulation with bacterial **lipopolysaccharide (LPS)**"; results: BCG reduced systemic inflammation, 75 genes with altered LPS response including CCL3/CCL4 with heightened response; co-expression modules lost coordination post-BCG while humanin nuclear isoforms gained coordination. Open. — [GEO GSE184241](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184241)
- **GSE248728** (**scRNA-seq**, `n=28`) and **GSE248720** (**scATAC-seq**, `n=14`), human, under SuperSeries **GSE248730** (`n=42`, PMID 39153479): BCG vaccination and the epigenetic landscape of human bone-marrow progenitor cells. Open. — [GEO GSE248730](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248730)
- **300BCG scRNA-seq**: the 300BCG portal lists single-cell RNA-seq QC statistics and gene-expression results "with and without LPS stimulation" as part of the released resource. — [300BCG portal](https://medical-epigenomics.org/papers/300bcg/)
- **GSE152353** (mouse, `n=4`, **scATAC-seq** of GMP and neutrophils from β-glucan-treated vs control mice, PMID 33125892). — [GEO GSE152353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152353)
- **GSE212092** (human scRNA-seq, `n=18`, "Systemic Inflammation Impairs Human Myelopoiesis"). — [GEO GSE212093](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212093)
- **Mouse single-cell trained-immunity sets (2025–2026)**: **GSE331226** (scRNA-seq, `n=4`), **GSE331085** (**CUT&Tag**, `n=3`), **GSE331083** (bulk RNA-seq, `n=6`) — "Metabolic–Epigenetic Rewiring of CCR5-hi Monocytes Sustains Long-Term Trained Immunity Against Lethal Sepsis"; **GSE235597** (scRNA-seq, `n=8`), **GSE235364** (**scMultiome**, `n=4`), **GSE235598** (ChIP-type, `n=12`) — "Systemically inducing trained immunity overcomes solid tumors' immunosuppressive microenvironment". — [GEO GSE331226](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE331226)
- **Non-human primate**: **GSE279314** (`n=14`, *Macaca mulatta*, **scMultiome**) and **GSE279313** (`n=110`, bulk RNA-seq): "TGF-β controls epigenetic re-programing of IFN-antiviral trained immunity and impairs virologic control in SIV-infected rhesus macaques" (2026). — [GEO GSE279313](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279313)

### Inferences
- There is no published human single-cell atlas of **β-glucan**-trained monocytes that I could find; human β-glucan work remains bulk RNA-seq/ChIP-seq, while single-cell β-glucan work is in mouse bone marrow. This is a real hole for anyone wanting per-cell β-glucan training states in human.

### Gaps
- GSE184241's `n=18` is a sample (library) count, not a donor count; donor numbers must be read from the paper (PMID 34788625).

---

## Key Question 6 — Metabolomics, proteomics, cytokine, flow/CyTOF and Seahorse data; and which datasets are matched multi-modal on the SAME donors

### Takeaway
Matched same-donor multi-modality is the strong suit of the Netea/Stunnenberg GEO SuperSeries and of 300BCG; it is the weak spot for metabolomics, proteomics, flow/CyTOF and Seahorse, where I found only a handful of relevant public deposits (2 Metabolomics Workbench studies, ~4 PRIDE projects, zero verified FlowRepository/ImmPort trained-immunity sets).

### Cited Findings — Metabolomics
- Metabolomics Workbench **ST003833**, *Homo sapiens*: "Lactate activates trained immunity by fueling the tricarboxylic acid cycle and regulating histone lactylation". — [Metabolomics Workbench REST, study_title "trained immunity"](https://www.metabolomicsworkbench.org/rest/study/study_title/trained%20immunity/summary)
- Metabolomics Workbench **ST004061**, *Mus musculus*: "In Vitro Method Demonstrating Trained Immunity as a Distinctive Functional Program: Implications for Biomarker [discovery]". This pairs with GEO **GSE290033** (RNA-seq, `n=11`, mouse) and **GSE290032** (ChIP/CUT&Tag-type, `n=17`, mouse), both titled "An in vitro methodology demonstrates the five steps of trained immunity in mice: implications on biomarker discovery, adaptive immune responses, mouse strains…" (released 2026/01/07) — i.e. **matched transcriptome + epigenome + metabolome on the same experimental system**. — [Metabolomics Workbench ST004061](https://www.metabolomicsworkbench.org/rest/study/study_title/trained%20immunity/summary); [GEO GSE290033](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290033)

### Cited Findings — Proteomics (PRIDE)
- **PXD000642**, *Homo sapiens*: "Quantitative proteomics of extracellular vesicles released from human primary macrophages upon beta/glucan sti[mulation]". — [PRIDE v3 API search "beta-glucan"](https://www.ebi.ac.uk/pride/archive/projects/PXD000642)
- **PXD045971**, *Homo sapiens*: "A chromatin-regulated biphasic circuit coordinates IL1B-mediated inflammation" — the proteomics companion to GEO **GSE244833** (`n=4`, human RNA-seq, same title, Mhlanga-lab linked). — [PRIDE PXD045971](https://www.ebi.ac.uk/pride/archive/projects/PXD045971)
- **PXD045072**, *Homo sapiens*: "Helicobacter pylori induces a novel form of **innate immune memory** via accumulation of NF-κB proteins". — [PRIDE PXD045072](https://www.ebi.ac.uk/pride/archive/projects/PXD045072)
- **PXD070363**, *Mus musculus*: "Trained to respond: Signal-level modifications of endosomal TLRs". — [PRIDE PXD070363](https://www.ebi.ac.uk/pride/archive/projects/PXD070363)
- **PXD015780**, *Homo sapiens*: "Immune modulation by complement receptor 3 (CD18/CD11b) dependent human monocyte TGF-β1-transporting vesicles" (adjacent, monocyte vesicle proteome). — [PRIDE PXD015780](https://www.ebi.ac.uk/pride/archive/projects/PXD015780)

### Cited Findings — Matched multi-modal on the SAME donors (highest value for a multi-omic model)
- **GSE58310** (Saeed 2014): single series combining RNA-seq and histone ChIP-seq across naïve monocyte, macrophage, β-glucan-trained and LPS-tolerized states — same donor panel, `n=64`. — [GEO GSE58310](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE58310)
- **GSE85246 family** (Novakovic 2016): one time-course experiment yielding RNA-seq (GSE85243), ChIP-seq (GSE85245), ATAC-seq (GSE87218) and, per the series summary, WGBS — the densest single-experiment multi-omic trained/tolerant resource in human monocytes. — [GEO GSE85246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85246)
- **GSE104149** (Arts 2018): RNA-seq + ChIP-seq on the same in vivo BCG-vaccinated volunteers (pre/post). — [GEO GSE104149](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104149)
- **GSE124220 family** (Cirovic 2020): bone-marrow RNA-seq + PBMC RNA-seq + ATAC-seq from the same BCG-vaccinated donors. — [GEO GSE124220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124220)
- **GSE153729 family** (Moorlag 2020): neutrophil RNA-seq + H3K4me3 ChIP-seq, same donors pre/post BCG. — [GEO GSE153729](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153729)
- **300BCG** (Moorlag 2024): ATAC-seq + scRNA-seq + cytokine production capacity + circulating inflammatory markers + immune cell frequencies on the **same 323 individuals** at day 0/14/90 — the only large human cohort here with chromatin, transcriptome and functional cytokine readouts on identical donors. GEO GSE241092 (processed, open) / EGAS50000000090 (raw, controlled) / Zenodo 10.5281/zenodo.10288920 (phenotypes, open). — [300BCG portal](https://medical-epigenomics.org/papers/300bcg/)
- **GSE248730 family**: scRNA-seq + scATAC-seq on the same BCG-vaccinated human bone-marrow donors. — [GEO GSE248730](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248730)
- **GSE262096–GSE262099**: histone lactylation/acetylation/methylation + RNA-seq across in vivo BCG vaccination and in vitro BCG training, plus an LDH-inhibitor perturbation arm — same protocol family, Netea/Novakovic/Mhlanga. — [GEO GSE262096](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262096)
- **GSE172116**: transcriptome (coding + non-coding) + chromatin accessibility + epigenetic modifications in the same human macrophage acute-vs-tolerant design. — [GEO GSE172116](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE172116)
- **GSE159249 + GSE159227**: RNA-seq and DNA-methylation arrays from the same tolerance experiment (`n=12` each). — [GEO GSE159249](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159249)
- **GSE290032 + GSE290033 + ST004061**: epigenome + transcriptome + metabolome, same mouse in-vitro trained-immunity system. — [GEO GSE290032](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290032)
- **GSE228184** (SuperSeries, `n=27`, human, PMID 39178113) "Trained immunity is facilitated by T cell-induced CD40-TRAF6 signaling" = **GSE228183** (RNA-seq, `n=9`) + **GSE228182** (ChIP-seq, `n=18`), same donors. — [GEO GSE228184](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE228184)
- **GSE271218 + GSE271219**: "C1q reprograms innate immune memory" RNA-seq (`n=6`) + ChIP-seq (`n=8`), human. — [GEO GSE271219](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271219)
- **GSE280787 + GSE280903**: "Trained immunity suppression determines kidney allograft survival" RNA-seq (`n=12`) + ChIP (`n=24`), human. — [GEO GSE280903](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280903)
- **GSE137044 family** (controlled human malaria infection, long-term trained immunity): SuperSeries `n=24` = **GSE137043** (RNA-seq, `n=12`) + **GSE137042** (ChIP-seq, `n=12`), human, released 2020/12/21. — [GEO GSE137044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137044)
- **GSE267418 + GSE267477**: "Cytokine-induced memory-like responses in endothelial cells link chronic inflammation to vascular disease risk" RNA-seq (`n=35`) + ChIP-type (`n=34`), human (2025). — [GEO GSE267477](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267477)

### Cited Findings — additional recent (through 2026) datasets relevant to training/activation states
- **GSE323311**, `n=48`, *Homo sapiens*, "Genome binding/occupancy profiling by high throughput sequencing", released 2026/03/26, PMID 42320478: "Human Macrophages Encode Stimulus-Specific Information of Prior Exposures Through Trained Immunity" — primary human monocyte-derived macrophages, **phenotypic and molecular stimulus specificity assessed six days after initial exposure**, "differential licensing of inflammatory transcription factors was associated with encoding of specificities in chromatin six days after training, while memory of some, but not all, training stimuli is lost by **eleven days** post-training in vitro." Open. — [GEO GSE323311](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE323311)
- **GSE242947**, `n=24`, *Homo sapiens*, RNA-seq, PMID 38228717: "**Dectin-1 ligands produce distinct training phenotypes** in human monocytes through differential activation of signaling networks" — compares multiple Dectin-1 agonists against *Candida albicans* β-glucan; important for signature specificity. Open. — [GEO GSE242947](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242947)
- **GSE290317**, `n=15`, human RNA-seq (2026/05/27): "Immune hyperactivity in hemodialysis patients is associated with **interferon gamma-induced trained immunity**". — [GEO GSE290317](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290317)
- **GSE289434**, `n=28`, *Mus musculus; Homo sapiens*, RNA-seq (2026/10/09): "Angiotensin II-induced hypertension induces **innate immune memory** and promotes atherosclerosis progression". — [GEO GSE289434](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE289434)
- **GSE335257**, mouse CUT&Tag-type (`n=6`, 2026/06/24): "Trained immunity links hematopoietic stem cell aging to aging-associated inflammation"; companion SIRT3 series **GSE307123** (ATAC-seq, `n=6`), **GSE307126/GSE307434** (RNA-seq), **GSE307432** (CUT&Tag, `n=4`). — [GEO GSE335257](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335257)
- **GSE141656**, `n=12`, human RNA-seq: "Transcriptomes of human **trained macrophages before and after Mtb exposure**". — [GEO GSE141656](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE141656)
- **GSE160184**, `n=12`, human RNA-seq: "Chronic HIV infection induces transcriptional and functional reprogramming of innate immune cells". — [GEO GSE160184](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE160184)
- **GSE235894** (human macrophage ChIP, `n=12`), **GSE235897** (human macrophage RNA-seq, `n=12`), **GSE216061** (mouse splenic macrophage ChIP, `n=9`), **GSE216062** (mouse spleen RNA-seq, `n=9`): adjuvant-induced trained immunity (protein-free vaccine). — [GEO GSE235894](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235894)
- **GSE270608**, `n=28`, mouse, ChIP-type: "High-Throughput Screen Identifies Non-Inflammatory Small Molecule Inducers of Trained Immunity" (2024). — [GEO GSE270608](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270608)
- **GSE285478** (human RNA-seq, `n=15`) + **GSE285851** (human lncRNA-seq, `n=12`): "Exosomal CLOCK-stabilizing lncRNA confers **innate memory in tuberculosis resisters**" (2025/11/27). — [GEO GSE285478](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE285478)
- **GSE240723** (human ChIP-seq, `n=20`) + **GSE301027** (mouse CUT&Run, `n=9`): "Hypoxia induces **histone clipping and H3K4me3 loss** in neutrophil progenitors resulting in long-term impairment of neutrophil immunity" (Mhlanga-linked, 2025). — [GEO GSE240723](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240723)

### Inferences
- If the model needs metabolomics *and* epigenomics on the same cells, the only verified fully-matched option is the mouse system (GSE290032/33 + ST004061); for human, the closest is 300BCG (chromatin + cytokine capacity + circulating markers on the same donors, but no metabolome on the portal's listed contents) or ST003833 (human lactate/histone-lactylation study, whose paired sequencing accession I did not verify).
- Proteomics is the thinnest modality: there is no public deep proteome of β-glucan- or BCG-trained human monocytes that I could verify, only an extracellular-vesicle proteome (PXD000642).

### Gaps
- **Seahorse / extracellular-flux (Agilent XF) data**: I found **no public repository deposit** for any trained-immunity study. These data appear only as figures. If needed, they would have to be requested from authors or digitised. No Zenodo DOI for Seahorse data was located.
- **Cytokine panels**: the only verified repository-hosted cytokine data are the 300BCG "cytokine production capacity" and "circulating inflammatory markers" files on Zenodo (10.5281/zenodo.10288920). HFGP/500FG cytokine QTL phenotypes are not in an open repository (only genotypes, controlled, EGAD00010002149).
- **Flow cytometry / CyTOF**: no FlowRepository FR-FCM or ImmPort SDY accession verified for human β-glucan or BCG trained immunity (searches returned nothing on-topic).
- **MetaboLights (MTBLS)**: the MetaboLights web-service search endpoint returned "There is no study" errors for all queries and the BioStudies MetaboLights collection returned 0 hits for "trained immunity", "macrophage polarization" and "monocyte LPS" — so I could not confirm *any* MTBLS accession for this field. Treat the absence as unverified rather than proven absent (the API behaviour looked like an endpoint problem, not necessarily an empty collection).
- **Hi-C**: no trained-immunity Hi-C dataset was verified. GEO searches for monocyte/macrophage Hi-C and promoter-capture returned only **GSE281909** (U937 4C-seq, `n=6`) and **GSE281908** (U937 ATAC-seq, `n=6`) from a CRISPRi regulatory-variant study, and **GSE136034** (`n=71`, human, "Corticosteroid-induced chromatin loop dynamics at the FKBP5 gene", Logie lab). The expected Fanucchi/Mhlanga chromatin-conformation accessions remain unverified (see Q1 gaps).
- **H3K9me3**: none of the verified series advertise H3K9me3 in their titles; the mark may be present inside the ChIP-seq subseries (GSE85245, GSE183528, GSE262096/97) but this needs a sample-level (GSM) check.
- **Donor counts and time points**: GEO `n_samples` is a library count. Donor numbers, priming durations, washout and rest intervals, and restimulation doses are not in the repository metadata for the great majority of these series and must be extracted from each paper's methods. The one record that states timing explicitly is GSE323311 (6-day and 11-day post-training assessments).
