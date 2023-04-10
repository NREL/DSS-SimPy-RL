from pyomo.environ import SolverFactory
import pyomo.environ as pe
import time

from ACPFR_Dist import *
import idaes

StartTime   = time.time()


ModelingTime = time.time() - StartTime
StartTime   = time.time()

instance = model.create_instance('ieee18.dat')  #To choose instance

instance.pprint()
#model.pprint()

ReadingTime = time.time() - StartTime
StartTime   = time.time()

opt= SolverFactory('ipopt')
results = opt.solve(instance, tee=True)

SolvingTime = time.time() - StartTime
StartTime   = time.time()

results.write()

print("------------------------------------------------------------------------------------------------------------")
print("-------------------------------SUMMARY----------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------------------------")
print("BUS RESULTS")
print("------------------------------------------------------------------------------------------------------------")
print("   Bus      V[pu]  Theta[Degree]     Pg[MW]     Qg[MVAr]       Pd[MW]     Qd[MVAr]      Gsh[MW]    Bsh[MVAr]")
print("------------------------------------------------------------------------------------------------------------")
for i in instance.BAR:
    a = (instance.bus_e[i].value ** 2 + instance.bus_f[i].value ** 2)
    b = 180 / 3.14159265359 * pe.atan((instance.bus_f[i].value) / instance.bus_e[i].value)
    print('%5d  %10.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f'
          % (i, a ** 0.5, b,
             instance.Sbase * instance.bus_Pg[i].value, instance.Sbase * instance.bus_Qg[i].value,
             instance.Sbase * instance.Pd_pu[i], instance.Sbase * instance.Qd_pu[i],
             instance.Sbase * instance.gshb[i] * a,
             instance.Sbase * instance.bshb[i] * a))

a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
for i in instance.BAR:
    a1 = (instance.bus_e[i].value ** 2 + instance.bus_f[i].value ** 2)
    a += instance.bus_Pg[i].value
    b += instance.bus_Qg[i].value
    c += instance.Pd[i]
    d += instance.Qd[i]
    e += instance.gshb[i] * a1
    f += instance.bshb[i] * a1
print(
    "------------------------------------------------------------------------------------------------------------")
print('TOTAL %37.4f %12.4f %12.4f %12.4f %12.4f %12.4f'
      % (instance.Sbase * a, instance.Sbase * b, c, d, instance.Sbase * e, instance.Sbase * f))
#
print("------------------------------------------------------------------------------------------------------------")
print("BRANCH RESULTS")
print("------------------------------------------------------------------------------------------------------------")
print("    i     j        Pij[MW]         Pji[MW]       Qij[MVAr]       Qji[MVAr]         Pls[MW]       Qls[MVAr]")
print("------------------------------------------------------------------------------------------------------------")
for i, j in instance.RAM:
    print('%5d %5d %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f'
          % (i, j,
             instance.Sbase * (instance.branch_Pfrom[i, j].value),
             instance.Sbase * instance.branch_Pto[i, j].value,
             instance.Sbase * (instance.branch_Qfrom[i, j].value),
             instance.Sbase * (instance.branch_Qto[i, j].value),
             ((instance.Sbase * (instance.branch_Pfrom[i, j].value + instance.branch_Pto[i, j].value)) ** 2) ** 0.5,
             ((instance.Sbase * (instance.branch_Qfrom[i, j].value + instance.branch_Qto[i, j].value)) ** 2) ** 0.5))

a = 0
b = 0
for i, j in instance.RAM:
    a = a + instance.branch_Pto[i, j].value + instance.branch_Pfrom[i, j].value
    b = b + instance.branch_Qto[i, j].value + instance.branch_Qfrom[i, j].value
print("------------------------------------------------------------------------------------------------------------")
print('TOTAL %85.4f %15.4f'
      % (instance.Sbase * a, instance.Sbase * b))
print("------------------------------------------------------------------------------------------------------------")

WritingTime = time.time() - StartTime
StartTime   = time.time()

print('Modeling               time', ModelingTime       )
print('Reading DATA           time', ReadingTime  )
print('Solving                time', SolvingTime )
print('Writing                time', WritingTime )