from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def rf():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )

def gbr():
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

try:
    from xgboost import XGBRegressor
    def xgb():
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
except:
    pass