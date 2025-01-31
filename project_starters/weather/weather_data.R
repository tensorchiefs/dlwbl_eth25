###### Collecting the meassured data
library(rdwd)
id = findID('Konstanz')
print(id)

KN_pos = selectDWD(id=id, res='hourly', var='', per='recent')
KN_recent = dataDWD(KN_pos, read = TRUE, varnames=TRUE)

akt_temp = KN_recent$hourly_air_temperature_recent_stundenwerte_TU_02712_akt
akt_temp$MESS_DATUM[1:4]

# Check the time difference between the rows is 0 hour
dfa = akt_temp#[1:100,]
for (i in 1:(nrow(dfa) -1)){
  delta = dfa[i+1,'MESS_DATUM'] - dfa[i,'MESS_DATUM']
  if (delta != 1){
    print("ACHTUNG")
  } 
}

lead_time = 24
max_lag = 48
features = c('TT_TU.Lufttemperatur', 'RF_TU.Relative_Feuchte')
target = 'TT_TU.Lufttemperatur'
N = nrow(akt_temp)
nrow = N-max_lag + 1 - lead_time
X = matrix(NA, ncol = length(features)*max_lag, nrow = nrow)
y = rep(NA, nrow = nrow)
times = rep(NA, nrow = nrow)
for (i in (max_lag+1):(N-lead_time + 1)){
  # i = (max_lag+1)
  y[i - max_lag] = dfa[i + lead_time - 1, target]
  times[i - max_lag] = dfa[i + lead_time - 1,'MESS_DATUM']
  j = 0
  for (f in features) {
    val = dfa[(i-max_lag):(i-1),f]
    X[i - max_lag, (j*max_lag+1):(j*max_lag+max_lag)] = val
    j = j + 1
  }
}
X 
lags = paste0("lag_", max_lag:1)
colnames(X) =  paste(rep(features, each=max_lag), rep(lags, length(features)))

library(readr)
# Save X and Y
saveRDS(list(X,y, times), file='~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/DL_WBL/HS24_25_WBL/Projekte/wetter/DWD_lead_time24_max_lag48.rds')
write_csv(data.frame(X), file='~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/DL_WBL/HS24_25_WBL/Projekte/wetter/DWD_lead_time24_max_lag48_X.csv')
write_csv(data.frame(y), file='~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/DL_WBL/HS24_25_WBL/Projekte/wetter/DWD_lead_time24_max_lag48_y.csv')
write_csv(data.frame(times), file='~/Dropbox/__ZHAW/__Projekte_Post_ZHAH/DL_WBL/HS24_25_WBL/Projekte/wetter/DWD_lead_time24_max_lag48_Times.csv')

#### Some initial Analysis
df_x = data.frame(X[1:1000,])
df_x$y = y[1:1000]
names(df_x)
rf = randomForest::randomForest(y ~ ., df_x,importance=TRUE)
#rf5 = rf
randomForest::varImpPlot(rf)
print(rf)
plot(df_x$y, predict(rf))
abline(0,1, col='red')
sqrt(mean((df_x$y-predict(rf))^2))








