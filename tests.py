"""
Problema Primal-Dual: Planificación de Producción
Resuelve tanto el problema primal como el dual usando PuLP
"""

from pulp import *
import numpy as np

print("="*70)
print("PROBLEMA PRIMAL-DUAL: PLANIFICACIÓN DE PRODUCCIÓN")
print("="*70)

# ============================================================================
# PROBLEMA PRIMAL
# ============================================================================
print("\n" + "="*70)
print("RESOLVIENDO PROBLEMA PRIMAL")
print("="*70)

# Crear el problema de maximización
primal = LpProblem("Produccion_Primal", LpMaximize)

# Variables de decisión (cantidad a producir de cada producto)
x_A = LpVariable("x_A", lowBound=0, cat='Continuous')
x_B = LpVariable("x_B", lowBound=0, cat='Continuous')
x_C = LpVariable("x_C", lowBound=0, cat='Continuous')
x_D = LpVariable("x_D", lowBound=0, cat='Continuous')

# Función objetivo: Maximizar beneficio
primal += 40*x_A + 60*x_B + 50*x_C + 35*x_D, "Beneficio_Total"

# Restricciones de recursos
primal += 2*x_A + 4*x_B + 3*x_C + 1*x_D <= 200, "Restriccion_Maquinas"
primal += 3*x_A + 2*x_B + 5*x_C + 4*x_D <= 300, "Restriccion_Mano_Obra"
primal += 1*x_A + 3*x_B + 2*x_C + 4*x_D <= 250, "Restriccion_Materia_Prima"

# Restricciones adicionales
primal += x_A >= 10, "Demanda_Minima_A"
primal += x_B <= 30, "Demanda_Maxima_B"
primal += x_C >= 0.5*x_D, "Relacion_C_D"

# Resolver
primal.solve(PULP_CBC_CMD(msg=0))

# Mostrar resultados del primal
print(f"\nEstado de la solución: {LpStatus[primal.status]}")
print(f"\n{'SOLUCIÓN ÓPTIMA PRIMAL':^70}")
print("-"*70)
print(f"Producir producto A: {value(x_A):.2f} unidades")
print(f"Producir producto B: {value(x_B):.2f} unidades")
print(f"Producir producto C: {value(x_C):.2f} unidades")
print(f"Producir producto D: {value(x_D):.2f} unidades")
print(f"\n{'Beneficio máximo: ' + f'{value(primal.objective):.2f} €':^70}")

# Análisis de restricciones del primal
print(f"\n{'ANÁLISIS DE RECURSOS (PRIMAL)':^70}")
print("-"*70)

recursos_info = [
    ("Máquinas", 2*value(x_A) + 4*value(x_B) + 3*value(x_C) + 1*value(x_D), 200),
    ("Mano de obra", 3*value(x_A) + 2*value(x_B) + 5*value(x_C) + 4*value(x_D), 300),
    ("Materia prima", 1*value(x_A) + 3*value(x_B) + 2*value(x_C) + 4*value(x_D), 250)
]

for nombre, usado, disponible in recursos_info:
    holgura = disponible - usado
    utilizado_pct = (usado/disponible)*100
    activa = "✓ ACTIVA" if abs(holgura) < 0.01 else "  Inactiva"
    print(f"{nombre:15} | Usado: {usado:6.2f}/{disponible:3} | Holgura: {holgura:6.2f} | {utilizado_pct:5.1f}% | {activa}")

# Extraer precios sombra (valores duales) del primal
print(f"\n{'PRECIOS SOMBRA (VALORES DUALES)':^70}")
print("-"*70)
print("(Cuánto aumentaría el beneficio por cada unidad adicional del recurso)")
print()

for nombre, constraint in primal.constraints.items():
    precio_sombra = constraint.pi
    if precio_sombra is not None and abs(precio_sombra) > 0.001:
        print(f"{nombre:30} | Precio sombra: {precio_sombra:8.2f} €")

# ============================================================================
# PROBLEMA DUAL
# ============================================================================
print("\n" + "="*70)
print("RESOLVIENDO PROBLEMA DUAL")
print("="*70)

# Crear el problema dual (minimización)
dual = LpProblem("Produccion_Dual", LpMinimize)

# Variables duales (precios sombra de cada recurso/restricción)
y1 = LpVariable("y1_Maquinas", lowBound=0, cat='Continuous')
y2 = LpVariable("y2_Mano_Obra", lowBound=0, cat='Continuous')
y3 = LpVariable("y3_Materia_Prima", lowBound=0, cat='Continuous')
y4 = LpVariable("y4_Demanda_Min_A", lowBound=0, cat='Continuous')
y5 = LpVariable("y5_Demanda_Max_B", lowBound=0, cat='Continuous')
y6 = LpVariable("y6_Relacion_CD", lowBound=0, cat='Continuous')

# Función objetivo dual: Minimizar valor de los recursos
# Los coeficientes son los lados derechos de las restricciones del primal
dual += 200*y1 + 300*y2 + 250*y3 + 10*y4 + 30*y5 + 0*y6, "Costo_Recursos"

# Restricciones duales (una por cada variable del primal)
# Cada restricción viene de los coeficientes de una variable en el primal

# Para producto A: 2*y1 + 3*y2 + 1*y3 + 1*y4 >= 40
dual += 2*y1 + 3*y2 + 1*y3 + 1*y4 >= 40, "Restriccion_Producto_A"

# Para producto B: 4*y1 + 2*y2 + 3*y3 - 1*y5 >= 60
dual += 4*y1 + 2*y2 + 3*y3 - 1*y5 >= 60, "Restriccion_Producto_B"

# Para producto C: 3*y1 + 5*y2 + 2*y3 + 1*y6 >= 50
dual += 3*y1 + 5*y2 + 2*y3 + 1*y6 >= 50, "Restriccion_Producto_C"

# Para producto D: 1*y1 + 4*y2 + 4*y3 - 0.5*y6 >= 35
dual += 1*y1 + 4*y2 + 4*y3 - 0.5*y6 >= 35, "Restriccion_Producto_D"

# Resolver
dual.solve(PULP_CBC_CMD(msg=0))

# Mostrar resultados del dual
print(f"\nEstado de la solución: {LpStatus[dual.status]}")
print(f"\n{'SOLUCIÓN ÓPTIMA DUAL':^70}")
print("-"*70)
print("Precios sombra óptimos (valor de cada restricción):")
print(f"  y1 (Máquinas):           {value(y1):.4f} €/hora")
print(f"  y2 (Mano de obra):       {value(y2):.4f} €/hora")
print(f"  y3 (Materia prima):      {value(y3):.4f} €/kg")
print(f"  y4 (Demanda mín A):      {value(y4):.4f} €/unidad")
print(f"  y5 (Demanda máx B):      {value(y5):.4f} €/unidad")
print(f"  y6 (Relación C-D):       {value(y6):.4f} €")

print(f"\n{'Costo mínimo de recursos: ' + f'{value(dual.objective):.2f} €':^70}")

# ============================================================================
# VERIFICACIÓN DE DUALIDAD
# ============================================================================
print("\n" + "="*70)
print("VERIFICACIÓN DE DUALIDAD FUERTE")
print("="*70)

z_primal = value(primal.objective)
w_dual = value(dual.objective)
diferencia = abs(z_primal - w_dual)

print(f"\nValor óptimo PRIMAL (beneficio máximo):  {z_primal:.4f} €")
print(f"Valor óptimo DUAL (costo mínimo):        {w_dual:.4f} €")
print(f"Diferencia:                              {diferencia:.6f} €")

if diferencia < 0.01:
    print("\n✓ DUALIDAD FUERTE VERIFICADA")
    print("  Los valores son iguales (dentro del error numérico)")
else:
    print("\n✗ ADVERTENCIA: Los valores difieren significativamente")

# ============================================================================
# CONDICIONES DE HOLGURA COMPLEMENTARIA
# ============================================================================
print("\n" + "="*70)
print("CONDICIONES DE HOLGURA COMPLEMENTARIA")
print("="*70)
print("\nPrincipio: Si una restricción primal tiene holgura > 0,")
print("           entonces su variable dual correspondiente debe ser 0")
print("           (y viceversa)")
print()

# Verificar para cada restricción
print(f"{'Restricción':30} | {'Holgura Primal':15} | {'Variable Dual':15} | {'Producto':10}")
print("-"*70)

# Restricción de máquinas
holgura_maq = 200 - (2*value(x_A) + 4*value(x_B) + 3*value(x_C) + 1*value(x_D))
producto_maq = holgura_maq * value(y1)
print(f"{'Máquinas':30} | {holgura_maq:15.4f} | {value(y1):15.4f} | {producto_maq:10.6f}")

# Restricción de mano de obra
holgura_mo = 300 - (3*value(x_A) + 2*value(x_B) + 5*value(x_C) + 4*value(x_D))
producto_mo = holgura_mo * value(y2)
print(f"{'Mano de obra':30} | {holgura_mo:15.4f} | {value(y2):15.4f} | {producto_mo:10.6f}")

# Restricción de materia prima
holgura_mp = 250 - (1*value(x_A) + 3*value(x_B) + 2*value(x_C) + 4*value(x_D))
producto_mp = holgura_mp * value(y3)
print(f"{'Materia prima':30} | {holgura_mp:15.4f} | {value(y3):15.4f} | {producto_mp:10.6f}")

print("\nSi el producto ≈ 0, la condición de holgura complementaria se cumple ✓")

# ============================================================================
# INTERPRETACIÓN ECONÓMICA
# ============================================================================
print("\n" + "="*70)
print("INTERPRETACIÓN ECONÓMICA")
print("="*70)

print("\n1. PRECIOS SOMBRA (del análisis dual):")
print("   Los precios sombra indican cuánto aumentaría el beneficio")
print("   si tuviéramos UNA UNIDAD MÁS de cada recurso:")
print()

if value(y1) > 0.001:
    print(f"   • Máquinas: {value(y1):.2f} €/hora adicional")
    print(f"     → Una hora más de máquina aumentaría el beneficio en {value(y1):.2f} €")

if value(y2) > 0.001:
    print(f"   • Mano de obra: {value(y2):.2f} €/hora adicional")
    print(f"     → Una hora más de trabajo aumentaría el beneficio en {value(y2):.2f} €")

if value(y3) > 0.001:
    print(f"   • Materia prima: {value(y3):.2f} €/kg adicional")
    print(f"     → Un kg más de materia prima aumentaría el beneficio en {value(y3):.2f} €")

print("\n2. DECISIONES DE PRODUCCIÓN:")
print(f"   • Producto más rentable: Produce {value(x_B):.0f} unidades de B")
print(f"   • La restricción de demanda máxima de B {'está activa' if abs(value(x_B) - 30) < 0.01 else 'no está activa'}")

print("\n3. RECURSOS CUELLOS DE BOTELLA:")
recursos_activos = []
if abs(holgura_maq) < 0.01:
    recursos_activos.append("Máquinas")
if abs(holgura_mo) < 0.01:
    recursos_activos.append("Mano de obra")
if abs(holgura_mp) < 0.01:
    recursos_activos.append("Materia prima")

if recursos_activos:
    print(f"   Los recursos que limitan la producción son: {', '.join(recursos_activos)}")
    print("   Estos recursos se usan al 100% de su capacidad")
else:
    print("   No hay recursos que limiten completamente la producción")

print("\n" + "="*70)
print("FIN DEL ANÁLISIS")
print("="*70)    