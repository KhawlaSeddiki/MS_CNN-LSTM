library(MSnbase)

# Create an intensity matrix from a list of MS spectra
# spectra_files must be in mzXML format
import_sp <- function(spectra_files, pas, tic_val) {
  # Use spectra only with MS level 1
  raw <- readMSData(spectra_files, msLevel=1)
  # Total ion count > tic_val
  raw <- raw[tic(raw)> tic_val]
  # RT
  rt <- rtime(chromatogram(raw)[1,1])
  # Bin mz values
  small_bined <- cbind(rt, do.call(rbind, MSnbase::intensity(bin(raw, binSize=1))))
  k=1
  bined <- list()
  for (j in seq(0, nrow(small_bined), by=pas)){
    if (length(small_bined[which(j < small_bined[,1] & small_bined[,1] < j+pas),-1]) > ncol(small_bined)-1){
      bined[[k]] <- colSums(small_bined[which(j < small_bined[,1] & small_bined[,1] < j+pas),-1])}
    else {bined[[k]] <- small_bined[which(j < small_bined[,1] & small_bined[,1] < j+pas),-1]}
    k=k+1}
  sp_large <- do.call(rbind, bined)
  return(sp_large)
}
