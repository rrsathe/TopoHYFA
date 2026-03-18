# Script to extract the 15 specific Ensembl IDs for target tissue and confounders
# for integration with HYFA PyTorch pipeline.
# To be sourced directly in regression_model_articleoutput.r after line 84

export_data_for_hyfa <- function(gtex_tss2, gtex_pc, confound_matrix, pat_com, outdir) {
  # 1. Target Ensembl IDs filter
  target_genes <- c(
    "ENSG00000016391", "ENSG00000072310", "ENSG00000118298", "ENSG00000134716", 
    "ENSG00000150281", "ENSG00000157734", "ENSG00000164089", "ENSG00000166317", 
    "ENSG00000175220", "ENSG00000178397", "ENSG00000188536", "ENSG00000197299", 
    "ENSG00000198517", "ENSG00000198830", "ENSG00000205208"
  )
  
  target_idx <- which(gtex_pc$Name %in% target_genes)
  
  if(length(target_idx) == 0) {
    warning("None of the target genes were found in the dataset.")
  }
  
  filtered_gtex_tss2 <- gtex_tss2[target_idx, , drop=FALSE]
  rownames(filtered_gtex_tss2) <- gtex_pc$Name[target_idx]
  colnames(filtered_gtex_tss2) <- pat_com
  
  # Transpose to Samples x Genes format for AnnData ingestion (t(matrix))
  target_genes_matrix <- t(filtered_gtex_tss2)
  
  # Force explicit Participant ID header natively upon export for seamless Pandas injection
  target_df <- data.frame(Participant_ID = rownames(target_genes_matrix), target_genes_matrix, check.names = FALSE)
  colnames(target_df)[1] <- "Participant ID"
  
  target_file <- paste0(outdir, "/target_genes_15.csv")
  write.csv(target_df, file = target_file, row.names = FALSE, quote = FALSE)
  print(paste("Data seamlessly exported. Target genes written to:", target_file))
  
  # 2. Confounders export aligned with the exact same mapping
  rownames(confound_matrix) <- pat_com
  confound_df <- data.frame(Participant_ID = rownames(confound_matrix), confound_matrix, check.names = FALSE)
  colnames(confound_df)[1] <- "Participant ID"
  
  confound_file <- paste0(outdir, "/confounders.csv")
  write.csv(confound_df, file = confound_file, row.names = FALSE, quote = FALSE)
  print(paste("Confounders written to:", confound_file))
}

# --------------------------------------------------------------------------------------
# Usage example assuming execution via base `regression_model_articleoutput.r` namespace:
# 
# outdir_hyfa <- paste0(workdir, "/output/HYFA_export")
# if(!file.exists(outdir_hyfa)) dir.create(outdir_hyfa, recursive = TRUE)
#
# export_data_for_hyfa(gtex.tss2, gtex.pc, confund, pat.com, outdir_hyfa)
# --------------------------------------------------------------------------------------
