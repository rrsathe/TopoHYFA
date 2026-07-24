#!/usr/bin/env Rscript
## =============================================================================
## 02_collect_data.R   [run order: 02]
## Pull the two HD cohorts from recount3:
##   SRP051844 -- BA9 frontal cortex, HD vs control   [tissue leg -> DEG (03) / DNB (04)]
##   SRP032279 -- whole blood, HD progression cohort  [blood leg  -> classifier, later]
## GTEx training data is HYFA's bundled GTEX_data.csv (already used in 02a); not re-pulled.
## Writes protein-coding symbol-level raw counts + phenotype (with HD/Control label) for each.
## RECORD: if it errors, fix and re-run top to bottom.
## =============================================================================

options(stringsAsFactors = FALSE)

## ---- config ----
STUDIES <- list(
  cortex = list(srp = "SRP051844", prefix = "hd_cortex_ba9"),
  blood  = list(srp = "SRP032279", prefix = "hd_blood")
)
KEEP_BIOTYPE <- "protein_coding"

## ---- project root + output dir (discovered) ----
find_root <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (!is.null(p) && nzchar(p)) return(normalizePath(dirname(p), winslash = "/"))
  }
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) && nzchar(f[1])) return(normalizePath(dirname(f[1]), winslash = "/"))
  normalizePath(getwd(), winslash = "/")
}
ROOT <- find_root(); OUT <- file.path(ROOT, "prep"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
message("project root: ", ROOT, "\noutput: ", OUT)

## ---- packages ----
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
for (pkg in c("recount3", "SummarizedExperiment", "edgeR", "AnnotationDbi", "org.Hs.eg.db"))
  if (!requireNamespace(pkg, quietly = TRUE)) BiocManager::install(pkg, update = FALSE, ask = FALSE)
suppressPackageStartupMessages({
  library(recount3); library(SummarizedExperiment); library(edgeR)
  library(AnnotationDbi); library(org.Hs.eg.db)
})

## ---- helpers (gene annotation / symbol dedup: same as the GTEx pull) ----
to_numeric_matrix <- function(x){ if(!is.matrix(x)) x<-as.matrix(x); storage.mode(x)<-"numeric"; x }
to_nonneg_int     <- function(x){ x<-round(x); x[is.na(x)]<-0; x[x<0]<-0; storage.mode(x)<-"integer"; x }
sanitize_ensembl  <- function(x) gsub("\\..*$","",as.character(x))
make_export_df    <- function(id_name, ids, mat)
  data.frame(setNames(list(ids), id_name), mat, check.names = FALSE, row.names = NULL)

get_gene_annotation <- function(rse, mat){
  rd <- as.data.frame(rowData(rse)); symbol <- NULL
  for (nm in c("gene_name","external_gene_name","hgnc_symbol","symbol"))
    if (nm %in% names(rd)){ x<-as.character(rd[[nm]]); if(any(!is.na(x)&nzchar(x))){symbol<-x; break} }
  ens <- sanitize_ensembl(rownames(mat))
  if (is.null(symbol)){
    map <- AnnotationDbi::select(org.Hs.eg.db, keys=unique(ens), columns="SYMBOL", keytype="ENSEMBL")
    map <- map[!duplicated(map$ENSEMBL),,drop=FALSE]; symbol <- map$SYMBOL[match(ens, map$ENSEMBL)]
  }
  biotype <- rep(NA_character_, nrow(mat))
  for (nm in c("gene_type","gene_biotype","biotype","type_of_gene"))
    if (nm %in% names(rd)){ biotype<-as.character(rd[[nm]]); break }
  data.frame(ensembl_id=rownames(mat), ensembl_noversion=ens,
             gene_symbol=as.character(symbol), gene_biotype=as.character(biotype))
}

pick_one_row_per_symbol <- function(mat_rank, annot, keep_biotype=NULL){
  sym<-annot$gene_symbol; bio<-annot$gene_biotype
  ok <- !is.na(sym) & nzchar(sym)
  if (!is.null(keep_biotype) && !all(is.na(bio))) ok <- ok & !is.na(bio) & (bio %in% keep_biotype)
  idx <- which(ok); if(!length(idx)) stop("no rows after symbol/biotype filter")
  grp <- split(idx, factor(sym[idx], levels=unique(sym[idx]))); us<-names(grp)
  keep <- integer(length(grp))
  for (i in seq_along(grp)){ ix<-grp[[i]]
    keep[i] <- if(length(ix)==1L) ix else ix[which.max(rowMeans(mat_rank[ix,,drop=FALSE], na.rm=TRUE))] }
  list(keep_ix=keep, keep_symbol=us)
}

## ---- SRA sample-attribute parsing ("k;;v|k;;v|...") ----
parse_all_attrs <- function(attr_vec){
  n <- length(attr_vec)
  sp <- strsplit(ifelse(is.na(attr_vec),"",as.character(attr_vec)), "\\|")
  kv <- lapply(sp, function(parts){
    parts <- parts[grepl(";;", parts, fixed=TRUE)]
    if(!length(parts)) return(list(k=character(0), v=character(0)))
    pieces <- strsplit(parts, ";;", fixed=TRUE)
    list(k = trimws(vapply(pieces, function(z) z[1], character(1))),
         v = trimws(vapply(pieces, function(z) paste(z[-1], collapse=";;"), character(1))))
  })
  allk <- unique(tolower(unlist(lapply(kv, `[[`, "k"))))
  if(!length(allk)) return(data.frame(row.names=seq_len(n)))
  df <- as.data.frame(matrix(NA_character_, n, length(allk)), stringsAsFactors=FALSE)
  colnames(df) <- paste0("attr_", make.names(allk))
  for (i in seq_len(n)){ ks<-tolower(kv[[i]]$k)
    if(length(ks)) df[i, match(ks, allk)] <- kv[[i]]$v }
  df
}

derive_condition <- function(attr_df, cd){
  cand <- c("diagnosis","disease","condition","phenotype","disease.state","disease_state",
            "subject.status","status","group","genotype","genotype.variation")
  cols <- colnames(attr_df); fld <- NULL
  for (k in cand){ c1<-paste0("attr_", make.names(k)); if(c1 %in% cols){ fld<-c1; break } }
  hay <- if(!is.null(fld)) attr_df[[fld]]
         else if ("sra.sample_attributes" %in% colnames(cd)) as.character(cd$sra.sample_attributes)
         else rep("", nrow(cd))
  is_hd  <- grepl("hunting|\\bHD\\b", hay, ignore.case=TRUE)
  is_ctl <- grepl("control|normal|unaffected|healthy|\\bctrl\\b", hay, ignore.case=TRUE) & !is_hd
  cond <- rep(NA_character_, length(hay)); cond[is_hd]<-"HD"; cond[is_ctl]<-"Control"
  list(field = if(is.null(fld)) "sra.sample_attributes(raw)" else fld, condition = cond)
}

## ---- ingest one SRA study ----
ingest <- function(srp, prefix){
  message("\n", strrep("=",60), "\n  ", srp, "  (", prefix, ")\n", strrep("=",60))
  proj <- available_projects()
  row  <- subset(proj, project==srp & file_source=="sra")
  if (nrow(row)==0L) stop(srp, " not found in recount3 (frozen ~Oct-2019 snapshot)")
  rse  <- create_rse(row)
  assay(rse,"counts") <- compute_read_counts(rse)
  message("  raw: ", nrow(rse), " genes x ", ncol(rse), " samples")

  annot <- get_gene_annotation(rse, assay(rse,"counts"))
  counts <- to_nonneg_int(to_numeric_matrix(assay(rse,"counts")))
  sel    <- pick_one_row_per_symbol(counts, annot, keep_biotype=KEEP_BIOTYPE)
  counts_sym <- counts[sel$keep_ix,,drop=FALSE]; rownames(counts_sym) <- sel$keep_symbol
  message("  protein-coding symbols: ", nrow(counts_sym))

  cd <- as.data.frame(colData(rse)); cd$sample_id <- colnames(rse)
  attr_df <- parse_all_attrs(if("sra.sample_attributes" %in% colnames(cd)) cd$sra.sample_attributes else NA)
  dc <- derive_condition(attr_df, cd)
  message("  condition field: ", dc$field,
          "  |  HD=", sum(dc$condition=="HD", na.rm=TRUE),
          "  Control=", sum(dc$condition=="Control", na.rm=TRUE),
          "  NA=", sum(is.na(dc$condition)))

  # write counts
  write.csv(make_export_df("gene_symbol", rownames(counts_sym), counts_sym),
            file.path(OUT, paste0(prefix,"_counts_by_symbol.csv")), row.names=FALSE)
  write.csv(make_export_df("ensembl_id", annot$ensembl_id, counts),
            file.path(OUT, paste0(prefix,"_counts_by_ensembl.csv")), row.names=FALSE)

  # phenotype (slim + full-with-attributes)
  pheno <- data.frame(sample_id = cd$sample_id, condition = dc$condition, attr_df, row.names=NULL)
  write.csv(pheno, file.path(OUT, paste0(prefix,"_pheno.csv")), row.names=FALSE)

  is_list <- vapply(cd, is.list, logical(1))
  if (any(is_list)) cd[is_list] <- lapply(cd[is_list], function(x)
    vapply(x, function(el) if(length(el)==0L||all(is.na(el))) "" else paste(unlist(el),collapse=","), character(1)))
  write.csv(cbind(cd, attr_df), file.path(OUT, paste0(prefix,"_pheno_full.csv")), row.names=FALSE, na="")

  message("  wrote ", prefix, "_{counts_by_symbol,counts_by_ensembl,pheno,pheno_full}.csv")
  invisible(list(n=ncol(rse), cond=table(dc$condition, useNA="ifany")))
}

## ---- run ----
recount3_cache <- tools::R_user_dir("recount3", which="cache")
dir.create(recount3_cache, recursive=TRUE, showWarnings=FALSE)
Sys.setenv("RECOUNT3_CACHE"=recount3_cache)
options(recount3_url = "http://duffel.rail.bio/recount3")

res_cortex <- ingest(STUDIES$cortex$srp, STUDIES$cortex$prefix)
res_blood  <- ingest(STUDIES$blood$srp,  STUDIES$blood$prefix)

## ---- manifest ----
manifest <- data.frame(
  leg    = c("tissue","blood"),
  srp    = c(STUDIES$cortex$srp, STUDIES$blood$srp),
  prefix = c(STUDIES$cortex$prefix, STUDIES$blood$prefix),
  n_samples = c(res_cortex$n, res_blood$n),
  counts_file = c(paste0(STUDIES$cortex$prefix,"_counts_by_symbol.csv"),
                  paste0(STUDIES$blood$prefix,"_counts_by_symbol.csv")),
  pheno_file  = c(paste0(STUDIES$cortex$prefix,"_pheno.csv"),
                  paste0(STUDIES$blood$prefix,"_pheno.csv"))
)
write.csv(manifest, file.path(OUT,"manifest.csv"), row.names=FALSE)

message("\n", strrep("=",60))
message("  COLLECTED. outputs in ", OUT)
message("  cortex (SRP051844): ", paste(names(res_cortex$cond), res_cortex$cond, sep="=", collapse="  "))
message("  blood  (SRP032279): ", paste(names(res_blood$cond),  res_blood$cond,  sep="=", collapse="  "))
message("  -> cortex feeds 03_deg_tissue.R (HD vs Control DEGs for netcontrol).")
message("  -> blood labels: eyeball hd_blood_pheno.csv against hd_blood_pheno_full.csv before the classifier step.")
message(strrep("=",60))
