from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class PredictorDegradacion:
    def __init__(self):
        # Pipeline: Expansión Cuadrática -> Escalado -> Ridge Regression
        # Permitimos coeficientes negativos para capturar el 'dip' del warm-up
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=0.1))
        ])
        self.compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
        self.offset_calibracion = 0.0
        
    def entrenar(self, df, track_base_time=81.5):
        # Mapear compuestos a valores numéricos
        df = df.copy()
        df['Compound_Idx'] = df['Compound'].map(self.compound_map)
        
        X = df[['TyreLife', 'TrackTemp', 'Compound_Idx']]
        y = df['LapTime']
        
        self.model.fit(X, y)
        
        # ANCLAJE DE INTERCEPT (Calibración Estricta al Peak Grip en Vuelta 3)
        # Buscamos el tiempo en la vuelta 3 que es donde ahora forzamos el pico
        t_peak_pred = self.predecir_lap(3, 100, 'SOFT', track_name='Monza', apply_offset=False)
        self.offset_calibracion = track_base_time - t_peak_pred
        
        print(f"Modelo calibrado al Peak Grip. Desfase corregido: {self.offset_calibracion:.3f}s")
        
    def predecir_lap(self, tyre_life, temp_rk4, compound_name, track_name='Silverstone', apply_offset=True):
        compound_idx = self.compound_map.get(compound_name, 1)
        X_input = pd.DataFrame([[tyre_life, temp_rk4, compound_idx]], 
                              columns=['TyreLife', 'TrackTemp', 'Compound_Idx'])
        
        # 1. PACE OFFSET INICIAL (Sincronizado con el compuesto)
        pace_offsets = {'SOFT': -0.8, 'MEDIUM': 0.0, 'HARD': 0.5}
        current_pace_offset = pace_offsets.get(compound_name, 0.0)
        
        # Predicción base del modelo
        pred_base = self.model.predict(X_input)[0] + current_pace_offset
        
        # Obtener física del compuesto Y PISTA para degradación dinámica
        from extractor_datos import COMPOUNDS_PHYSICS, CIRCUITOS_CONFIG
        phys = COMPOUNDS_PHYSICS.get(compound_name, {'max_life': 40})
        m_life_base = phys['max_life']
        
        # ESCALADO POR PISTA 
        track_abrasion = CIRCUITOS_CONFIG.get(track_name, {'abrasion': 1.0})['abrasion']
        m_life_efectiva = m_life_base * (1.0 / track_abrasion)
        
        # 2. MULTIPLICADORES DE DEGRADACIÓN
        deg_multipliers = {'SOFT': 1.8, 'MEDIUM': 1.0, 'HARD': 0.4}
        mult = deg_multipliers.get(compound_name, 1.0)
        
        # 3. Warm-up Penalty (Out-lap lenta)
        warmup_penalty = 0
        if tyre_life < 4:
            warmup_penalty = 1.6 * np.exp(-(tyre_life - 0.8) / 0.8)
        
        # 4. Peak Grip Window
        peak_shape = 0.05 * (max(0, 4 - tyre_life)**2)
        
        # 5. Degradación cuadrática DINÁMICA (Escalada por m_life_efectiva)
        # Paracaídas Anti-Overflow: Límite máximo de 20s de pérdida
        deg_dinamica = 0
        if tyre_life > 5:
            # d y k_efectivo calibrados para el modelo físico
            d = phys.get('desgaste_base', 0.05) * mult
            k_efectivo = 0.18 * track_abrasion
            deg_dinamica = min(d * np.exp(k_efectivo * (tyre_life - 5)), 20.0)
            
        return pred_base + (self.offset_calibracion if apply_offset else 0) + warmup_penalty + peak_shape + deg_dinamica
