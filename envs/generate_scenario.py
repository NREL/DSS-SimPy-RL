# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 19:42:05 2022

@author: abhijeetsahu
"""

# idea is to generate sub-optimal scenarios for state space and prior action selection before feeding into the IRL learning problem

# read the open-dss 123 bus scenario
import sys
sys.path.append('../')
import os
directory = os.path.dirname(os.path.realpath(__file__))
desktop_path = os.path.dirname(os.path.dirname(directory))
sys.path.append(desktop_path+'\ARM_IRL')
import opendssdirect as dss
import os
import math
import csv
import re
import itertools
import pandas as pd
import numpy as np
from envs.resilience_graphtheory import GraphResilienceMetric

# function to extract transmission line and switch information
def get_lines(dss):
    Line = []
    Switch = []
    lines = dss.Lines.First()
    while lines:
        datum = {}
        line = dss.Lines
        datum["name"] = line.Name()
        datum["bus1"] = line.Bus1()
        datum["bus2"] = line.Bus2()
        if 'sw' in datum["name"]:
            datum["switch_flag"] = True
        else:
            datum["switch_flag"] = False
        #datum["switch_flag"] = dss.run_command('? Line.' + datum["name"] + '.Switch')
        if datum["switch_flag"] == False:
            datum["wires"] = dss.run_command('? Line.' + datum["name"] + '.Wires')
            datum["length"] = line.Length()
            datum['units'] = line.Units()
            datum["phases"] = line.Phases()
            datum["spacing"] = line.Spacing()
            datum["linecode"] = line.LineCode()
            datum["normAmp"] = line.NormAmps()
            datum["geometry"] = line.Geometry()
            #datum["R1"] = line.R1()*line.Length()
            #datum["X1"] = line.X1()*line.Length()
            datum["RMatrix"] = [x*line.Length() for x in line.RMatrix()]
            datum["XMatrix"] = [x*line.Length() for x in line.XMatrix()]
            Line.append(datum)
        else:
            Switch.append(datum)
        lines = dss.Lines.Next()
    return [Line, Switch]

# function to get transformer related information
def get_transformer(dss, circuit):
    data = []
    circuit.SetActiveClass('Transformer')
    xfmr_index = dss.ActiveClass.First()
    while xfmr_index:
        cktElement = dss.CktElement
        xfmr_name = cktElement.Name()
        buses = cktElement.BusNames()
        conns = dss.run_command('? ' + xfmr_name + '.conns')
        kVs = dss.run_command('? ' + xfmr_name + '.kVs')
        kVAs = dss.run_command('? ' + xfmr_name + '.kVAs')
        phase = dss.run_command('? ' + xfmr_name + '.phases')
        loadloss = dss.run_command('? ' + xfmr_name + '.%loadloss')
        noloadloss = dss.run_command('? ' + xfmr_name + '.%noloadloss')
        Rs = dss.run_command('? ' + xfmr_name + '.%Rs')
        xhl = dss.run_command('? ' + xfmr_name + '.xhl')
        dataline = dict(name=xfmr_name, buses=buses, conns=conns, kVs=kVs, kVAs=kVAs, phase=phase, loadloss=loadloss,
                        noloadloss=noloadloss, Rs=Rs, xhl=xhl)
        data.append(dataline)
        xfmr_index = dss.ActiveClass.Next()
    return data

# get the load values
def get_loads(dss, circuit, loadshape_flag=0, loadshape_folder=None):
    """ This function gets the list of load

    """
    data = []
    load_flag = dss.Loads.First()
    total_load = 0

    while load_flag:
        load = dss.Loads
        datum = {
            "name": load.Name(),
            "kV": load.kV(),
            "kW": load.kW(),
            "PF": load.PF(),
            "Delta_conn": load.IsDelta()
        }
        indexCktElement = circuit.SetActiveElement("Load.%s" % datum["name"])
        cktElement = dss.CktElement
        bus = cktElement.BusNames()[0].split(".")
        datum["kVar"] = float(datum["kW"]) / float(datum["PF"]) * math.sqrt(1 - float(datum["PF"]) * float(datum["PF"]))
        datum["bus1"] = bus[0]
        datum["numPhases"] = len(bus[1:])
        datum["phases"] = bus[1:]
        if not datum["numPhases"]:
            datum["numPhases"] = 3
            datum["phases"] = ['1', '2', '3']
        datum["voltageMag"] = cktElement.VoltagesMagAng()[0]
        datum["voltageAng"] = cktElement.VoltagesMagAng()[1]
        datum["power"] = dss.CktElement.Powers()[0:2]
        data.append(datum)
        load_flag = dss.Loads.Next()
        total_load += datum["kW"]

        if loadshape_flag == 1:  # read loadshape file to get 1-year data
            filename = os.path.join(loadshape_folder, datum["name"] + '_loadshape.csv')
            data0 = []
            with open(filename, 'r') as f:
                csvread = csv.reader(f)
                for row in csvread:
                    data0.append(float(row[0]))
                f.close()
            datum["1year_loadshape"] = data0

    return [data, total_load]

# get the voltages by bus names
def get_Vbus(dss, circuit, busname):  # busname doesn't has .1, .2, or .3
    """ This function gets the absolute voltage magnitude of the specified bus

        :param dss: openDss com object
        :type dss: object
        :param circuit: The circuit of the dss object. 
        :type circuit: dss.Circuit
        :param busname: bus name
        :type busname: str
        :return: voltage magnitude of the bus
        :rtype: float
    """
    circuit.SetActiveBus(busname)
    voltage = dss.Bus.VMagAngle()
    Vmag = [ii / dss.Bus.kVBase() / 1000 for ii in voltage[0:len(voltage):2]]
    # Vmag = [ii/1 for ii in voltage[0:len(voltage):2]]
    return Vmag

def getPowers(circuit, dss, type, names):
    """ This function gets the power consumed by a list of load buses

        :param dss: openDss com object
        :type dss: object
        :param circuit: The circuit of the dss object. 
        :type circuit: dss.Circuit
        :param type: circuit element type
        :type type: str
        :param names: list of load buses
        :type names: list
        :return: List of power consumed at the critical load buses
        :rtype: list
    """
    d = [None] * len(names)
    count = 0
    for loadname in names:
        if len(loadname.split('.')) > 1:
            circuit.SetActiveElement(loadname)
        else:
            circuit.SetActiveElement(type + '.' + loadname)
        s = dss.CktElement.Powers()
        d[count] = [sum(s[0:len(s):2]), sum(s[1:len(s):2])]
        count = count + 1
    d = np.asarray(d)
    powers = sum(d)
    return [d, powers]

# Have to test the function whether the CLOSE/OPEN for this works or not
# function to open all the sectional switch at the beginning
def open_switch_all(dss, circuit, switches_to_open):
    """ This function opens all the sectionalizing switches before the start of the episode 
    
        :param dss: openDss com object
        :type dss: object
        :param circuit: The circuit of the dss object. 
        :type circuit: dss.Circuit
        :param switches_to_open: List of switches to open
        :type switches_to_open: list
    """
    
    switches = dss.SwtControls.AllNames()
    #print(switches)
    for sw_Open in switches_to_open:
        circuit.SetActiveElement("Line.%s" % sw_Open)
        
        
        #dss.SwtControls.Name = sw_Open
        #dss.SwtControls.Action = 1
        #dss.SwtControls.IsLocked(True)
        dss.CktElement.Open(1,0)
    dss.Solution.SolvePlusControl()
        #print('Status : '+str(dss.CktElement.IsOpen(1,1)))

def close_switch(dss, circuit, switches_to_close, rest_switch,critical_load_buses):
    """ This function closes the set of sectionalizing switches and returns the voltage statuses of the critical loads

        :param dss: openDss com object
        :type dss: object
        :param circuit: The circuit of the dss object. 
        :type circuit: dss.Circuit
        :param switches_to_close: List of switches to close
        :type switches_to_close: list
        :param rest_switch: complete list of switches
        :type rest_switch: list
        :param critical_load_buses: List of critical load buses
        :type critical_load_buses: list
        :return: List of voltage status of critical load bus
        :rtype: list
    """
    for sw in rest_switch:
        circuit.SetActiveElement("Line.%s" % sw)
        if sw in switches_to_close:
            # the first argument is the terminal, and the second argument is the phase where 0: all 3 phase, 1,2,3 are phase a,b, and c
            dss.CktElement.Close(1,0)
        else:
            dss.CktElement.Open(1,0)
    dss.Solution.SolvePlusControl()
    volt_critical_loads=[]
    for cl in critical_load_buses:
        volt_critical_loads.append(get_Vbus(dss, circuit,cl))
    #print (' voltages when all {} closed : {}'.format(switches_to_close, volt_critical_loads))
        #print('Status : '+str(dss.CktElement.IsOpen(1,1)))

def close_one_switch(dss,circuit, sw_to_close, critical_load_buses):
    """ This function closes one switch

        :param dss: openDss com object
        :type dss: object
        :param circuit: The circuit of the dss object. 
        :type circuit: dss.Circuit
        :param sw_to_close: switch to close
        :type sw_to_close: str
        :param critical_load_buses: List of critical load buses
        :type critical_load_buses: list
        :return: List of voltage status of critical load bus
        :rtype: list
    """
    circuit.SetActiveElement("Line.%s" % sw_to_close)
    dss.CktElement.Close(1,0)
    dss.Solution.SolvePlusControl()
    volt_critical_loads=[]
    for cl in critical_load_buses:
        volt_critical_loads.append(get_Vbus(dss, circuit,cl))
    #print (' voltages when {} closed : {}'.format(sw_to_close, volt_critical_loads))


def close_switch_sequentially(dss, circuit, switches_to_close, rest_switch,critical_load_buses):
    """ This function sequentially closes the switch

        :param dss: openDss com object
        :type dss: object
        :param circuit: The circuit of the dss object. 
        :type circuit: dss.Circuit
        :param switches_to_close: List of switches to close
        :type switches_to_close: list
        :param rest_switch: complete list of switches
        :type rest_switch: list
        :param critical_load_buses: List of critical load buses
        :type critical_load_buses: list
        :return: List of voltage status of critical load bus
        :rtype: list
    """
    for sw in rest_switch:
        circuit.SetActiveElement("Line.%s" % sw)
        if sw in switches_to_close:
            # the first argument is the terminal, and the second argument is the phase where 0: all 3 phase, 1,2,3 are phase a,b, and c
            dss.CktElement.Close(1,0)
            dss.Solution.SolvePlusControl()
            volt_critical_loads=[]
            for cl in critical_load_buses:
                volt_critical_loads.append(get_Vbus(dss, circuit,cl))
            #print (' voltages when {} closed : {}'.format(sw,volt_critical_loads))
        else:
            dss.CktElement.Open(1,0)
    #dss.Solution.SolvePlusControl()
    
def disable_switches(dss, circuit, switches_to_open):
    """ This function disable the switches and run the power flow

    """
    for sw_Open in switches_to_open:
        dss.run_command('Disable Line.'+str(sw_Open))
    dss.run_command('solve')

def enable_switches(dss, circuit, switches_to_close):
    """ This function enable the switches and run the power flow

    """
    for sw_Close in switches_to_close:
        dss.run_command('Enable Line.'+str(sw_Close))
    dss.run_command('solve')

    
# cause line fault
def cause_line_fault(dss, line_number):
    """ This function causes the line outage.

        :param dss: openDss com object
        :type dss: object
        :param line_number: The transmission line that encounters outage
        :type line_number: str
        :return: List of voltage status of all the buses in the distribution feeder
        :rtype: list
    """
    dss.run_command('Disable Line.'+str(line_number))
    dss.run_command('solve')
    power_file = dss.run_command('export elempowers')
    voltage_file = dss.run_command('export voltages')
    dss_node_voltage = read_dss_result(power_file, voltage_file)
    return dss_node_voltage

# cause capacitor bank to open, for close change to 1
def cb_outage(dss, cb_name):
    """ This function implements the capacitor bank outage and run the power flow

        :param cb_name: the name of the capacitor bank selected 
        :type dss: str
    """
    #dssCircuit = dss.ActiveCircuit
    dss.Circuit.SetActiveElement("Capacitor.%s" % cb_name)
    dss.Capacitors.Name = str(cb_name)
    dss.Capacitors.States = (0,)
    dss.run_command('solve')

def der_outage(dss, der_bus_number):
    """ This function implements the DER outages and run the power flow

        :param der_bus_number: the bus name of the DER which will encounter outage
        :type dss: str
    """
    for d in der_bus_number:
        dss.run_command('Disable Generator.der'+str(der_bus_number)+'a')
        dss.run_command('Disable Generator.der'+str(der_bus_number)+'b')
        dss.run_command('Disable Generator.der'+str(der_bus_number)+'c')
    dss.run_command('solve')

def cb_restore(dss, cb_name):
    """ This function restores the capacitor banks and re-run the power flow

        :param cb_name: the name of the capacitor bank selected 
        :type dss: str
    """
    #dssCircuit = dss.ActiveCircuit
    dss.Circuit.SetActiveElement("Capacitor.%s" % cb_name)
    dss.Capacitors.Name = str(cb_name)
    dss.Capacitors.States = (1,)
    dss.run_command('solve')

def read_dss_result(power_file, voltage_file, bus_number=130):
    """ This function reads the dss results mainly V and P, obtained after running the power flow

        :param power_file: file path to the power output
        :type power_file: str
        :param voltage_file: file path to the voltage output
        :type voltage_file: str
    """
    voltage_result = np.zeros((bus_number, 3))
    p_power_result = np.zeros((bus_number, 3))
    q_power_result = np.zeros((bus_number, 3))

    vv = pd.read_csv(voltage_file).values.tolist()
    for i in range(len(vv)):
        temp_bus = i+1
        if temp_bus <= bus_number:
            temp_phase1 = int(vv[i][2])
            temp_phase2 = int(vv[i][6])
            temp_phase3 = int(vv[i][10])
            if temp_phase1 > 0:
                voltage_result[temp_bus - 1][temp_phase1 - 1] = vv[i][5]
            if temp_phase2 > 0:
                voltage_result[temp_bus - 1][temp_phase2 - 1] = vv[i][9]
            if temp_phase3 > 0:
                voltage_result[temp_bus - 1][temp_phase3 - 1] = vv[i][13]
    return voltage_result

# reset to original scenario

def voltage_satisfiability(volt_mag_list):
    """ Checks if the voltage are in limits 

        :param volt_mag_list: List of p.u. voltage at the buses
        :type volt_mag_list: list
        :return: True or Falsse depending on voltage satisfiability
        :type : bool
    """
    error = 0
    for v_mag in volt_mag_list:
        if abs(v_mag - 1.0) > 0.2:
            return False, 100.0
        else:
            error+=abs(v_mag - 1.0)
    return True, error/len(volt_mag_list)

def voltage_satisfiability_easy(volt_mag_list):
    error = 0
    for v_mag in volt_mag_list:
        if abs(v_mag) == 0:
            return False, 100.0
        else:
            error+=abs(v_mag - 1.0)
    return True, error/len(volt_mag_list)

def voltage_satisfiability_old(volt_mag_list):
    error = 0
    for v_mag in volt_mag_list:
        if abs(v_mag[0] - 1.0) > 0.2:
            return False, 100.0
        else:
            error+=abs(v_mag[0] - 1.0)
    return True, error/len(volt_mag_list)


# integrate load variation for more data collection

def randomize_load(dss,load_names,min_load=0.9,max_load=1.1):
    """ This function randomise the load profile. 
        
        :param load_names: List of load bus names
        :type load_names: list

    """
    MIN_BUS_VOLT = 0.9
    MAX_BUS_VOLT = 1.1
    while True:
        randScale = np.random.uniform(min_load, max_load, len(load_names))
        scale_up(dss, randScale,load_names)

        # after scaling the load values up, solve the pf and check if the voltages are in limits or it rescales down
        VoltageMag = dss.Circuit.AllBusMagPu()
        if (max(VoltageMag) > MAX_BUS_VOLT) & (min(VoltageMag) < MIN_BUS_VOLT):
            scale_down(dss, randScale,load_names)
        else:
            return


def scale_up(dss, randScale,load_names):
    """ This function scales up the load profile

        :param dss: openDss com object
        :type dss: object
        :param randScale: scaling factor
        :type randScale: float
        :param load_names: List of load bus names
        :type load_names: list
        :return: Nothing
        :type : None
    """
    # Step through every load and scale it up by a random percentage
    # iLoads = dss.Loads.First()
    rdx = 0
    for lname in load_names:
        dss.Circuit.SetActiveElement("Load.%s" % lname)
        dss.Loads.kW(dss.Loads.kW() * randScale[rdx])
        rdx+=1

def scale_down(dss, randScale,load_names):
    """ This function scales down the load profile

        :param dss: openDss com object
        :type dss: object
        :param randScale: scaling factor
        :type randScale: float
        :param load_names: List of load bus names
        :type load_names: list
        :return: Nothing
        :type : None
    """
    # Step through every load and scale back down by the same random percentage
    rdx = 0
    for lname in load_names:
        dss.Circuit.SetActiveElement("Load.%s" % lname)
        dss.Loads.kW(dss.Loads.kW() / randScale[rdx])
        rdx+=1


# This function compute the optimal scenarios, based on 3 criterias
def process_solution(dss,circuit,switch_names,critical_loads_bus, line_faults):
    """ This function prints the best solution based on various resilience metric. This function was written based on identifying expert demonstrations based on a defined resilience metric

        :param dss: openDss com object
        :type dss: object
        :param switch_names: List of sectionalizing switches in the feeder
        :type switch_names: list
        :param critical_loads_bus: List of critical load bus names
        :type critical_loads_bus: list
        :param line_faults: The list of line under outage
        :type line_faults: list
        :return: Nothing
        :type : None
    """
    feasible_solns = []
    loss_real_min = 100000
    loss_react_min = 100000
    min_switching = 7
    min_volt_error = 10000
    optimal_soln_min_switch = []
    optimal_soln_min_real_loss = []
    optimal_soln_min_react_loss = []
    optimal_soln_voltage = []

    for length in range(1,len(switch_names) + 1):
        for subset in itertools.combinations(switch_names, length):
            
            #close_switch(dss,circuit, subset,switch_names,critical_loads_bus)
            #enable_switches(dss,circuit, subset)

            close_switch_sequentially(dss,circuit, subset,switch_names,critical_loads_bus)
            
            #print ('Action attempt to Close : '+str(subset))
            # solve the case, obtain the voltage values of the critical nodes
            #dss.run_command('solve')
            #dss.Solution.Solve()

            # get the critical node voltages
            volt_critical_loads = []


            # convert the current circuit into networkx graph
            grm = GraphResilienceMetric(_dss = dss)

            # compute the resilience metric
            lf = (map(lambda x : x.lower(), line_faults))
            switch_status = (map(lambda x : x.lower(), subset))
            bcs = grm.compute_bc(list(lf),list(switch_status))
            cls = grm.compute_cl(list(lf),list(switch_status))
            ebcs = grm.compute_ebc(list(lf),list(switch_status))

            print('Average BC : {0}, Average CL : {1}, Average EBC : {2}'.format(str(np.mean(bcs)),str(np.mean(cls)),str(np.mean(ebcs)) ))

            for cl in critical_loads_bus:
                volt_critical_loads.append(get_Vbus(dss, circuit,cl))
            #print('Voltage Critical Loads : '+str(volt_critical_loads))

            satisfy_volt, v_err = voltage_satisfiability(volt_critical_loads)

            if satisfy_volt:
                feasible_solns.append(subset)
                Loss = dss.Circuit.LineLosses()
                if Loss[0] <= loss_real_min:
                    loss_real_min = Loss[0]
                    optimal_soln_min_real_loss = list(subset)
                if Loss[1] <= loss_react_min:
                    loss_react_min = Loss[1]
                    optimal_soln_min_react_loss = list(subset)
                if len(subset) <= min_switching:
                    min_switching = len(subset)
                    optimal_soln_min_switch = list(subset)
                if v_err <= min_volt_error:
                    min_volt_error = v_err
                    optimal_soln_voltage = list(subset)

            # you have to re-open all the switches then compute
            open_switch_all(dss,circuit, switch_names)


    print('*****************************************')
    print ('Optimal Soln by Min Real Power Loss '+str(optimal_soln_min_real_loss))
    print ('Optimal Soln by Min Reactive Power Loss '+str(optimal_soln_min_react_loss))
    print ('Optimal Soln by Min Switching '+str(optimal_soln_min_switch))
    print ('Optimal Soln by Min Voltage Deviation '+str(optimal_soln_voltage))
    print('*****************************************\n')

if __name__ == "__main__":

    global bus_number

    # Main code  
    dss_data_dir = desktop_path+'\\ARM_IRL\\cases\\123Bus_Simple\\'
    dss_master_file_dir = 'Redirect ' + dss_data_dir + 'IEEE123Master.dss'

    dss.run_command(dss_master_file_dir)
    circuit = dss.Circuit
    Bus_name_vec = circuit.AllBusNames()  # read bus name
    bus_number = len(Bus_name_vec)
    bus_vbase = []
    temp_bus_phases = []
    bus_phases = []

    # the following code derive bus phases, this might need modification when applied to other systems
    for i in range(bus_number):
        circuit.SetActiveBus(Bus_name_vec[i])
        bus_vbase.append(dss.Bus.kVBase())  # get bus base kV
        temp_bus_phases.append(dss.Bus.Nodes())

    for temp_index in range(bus_number):
        #temp_index = Bus_name_vec.index(str(i + 1))
        tempvec = [0, 0, 0]
        for j in range(len(temp_bus_phases[temp_index])):
            tempvec[temp_bus_phases[temp_index][j] - 1] = 1
        bus_phases.append(tempvec)

    line_info, switch_info = get_lines(dss)
    tran_info = get_transformer(dss, dss.Circuit)

    # currently we have a fixed load profile....
    load_info = []
    load_info, total_load_cap = get_loads(dss, dss.Circuit)

    # function to randomly modify load profile


    # list of contingencies to consider:???????????????????? For time being we consider random line faults
    # L55 : look for critical node at bus 59 or 58
    # L68 : look for critical node at either 99-100
    # L58 : look for critical node at node 88-96, also disable the capacitor bank as one of the contingencies
    # L77 : look for critical node at node say 78-84
    # L45 : look for critical node at node 48-51
    # L101 : look for critical node at 111-114
    # L41 : Look for critical node from 36-39

    line_faults = { 0: 'L55', 1: 'L68', 2: 'L58', 3: 'L77', 4: 'L45', 5: 'L101', 6:'L41'}
    #line_faults = { 0: 'L55', 1: 'L77'}

    cbs = {0: 'C83', 1: 'C88a', 2: 'C90b', 3: 'C92c'}

    # switch from and two buses, with the first 6 are normally closed and the last two are normally open
    switches = { 0: ['150r','149'], 1: ['13','152'], 2: ['18','135'], 3: ['60','160'], 4: ['97','197'], 5: ['61','61s'], 6: ['151','300'], 7: ['54','94'] }

    switch_names =[]
    for k,sw in enumerate(switches):
        switch_names.append('Sw'+str(k+1))

    # remove the switch to control that is connected to substation
    switch_names.remove('Sw1')

    # randomly define some critical loads in the IEEE 123 bus case
    critical_loads = ['S58b','S59b','S111a','S114a','s88a','S92c','S94a','S24c','S48','S50c']
    #critical_loads_bus = ['58','59','111','114','88','92','94','24','48','50']
    critical_loads_bus = ['58','59','99','100','88','93','94','78','48','50', '111','114', '37','39']

   
    #dss.run_command('solve')
    power_file = dss.run_command('export elempowers')
    voltage_file = dss.run_command('export voltages')
    dss_node_voltage = read_dss_result(power_file, voltage_file)

    data_gen_factor = 1000
    load_names =list(dss.Loads.AllNames())

    for r in range(data_gen_factor):

        # randomize load
        randomize_load(dss, load_names)

        # solve all N-1 contingency for line outage
        for i,(k,lf) in enumerate(line_faults.items()):

            # open all switches
            open_switch_all(dss,circuit, switch_names)
            #disable_switches(dss,circuit, switch_names)
            #dss.Solution.Solve()

            dss_node_voltage =  cause_line_fault(dss,str(lf))
            print('*****************************************')
            print ('Contingency : Line Fault '+str(lf))

            process_solution(dss,circuit,switch_names,critical_loads_bus,[lf])

            # restore contingency
            dss.run_command('Enable Line.'+str(lf))

            
        # N-2 outage a line fault and capacitor bank out
        for i,(k,lf) in enumerate(line_faults.items()):
            for j,(k1,cb) in enumerate(cbs.items()):
                # open all switches
                open_switch_all(dss,circuit, switch_names)
                #disable_switches(dss,circuit, switch_names)
                #dss.Solution.Solve()

                # N-2
                dss_node_voltage =  cause_line_fault(dss,str(lf))
                cb_outage(dss,str(cb))

                print('*****************************************')
                print ('Contingency : Line Fault '+str(lf)+' and Capacitor Bank out '+str(cb))
                
                process_solution(dss,circuit,switch_names,critical_loads_bus,[lf])

                # restore contingency
                dss.run_command('Enable Line.'+str(lf))
                cb_restore(dss,str(cb))


        # N-2 outage 2 line faults
        for i,(k,lf) in enumerate(line_faults.items()):
            for j,(k1,lf2) in enumerate(line_faults.items()):

                if lf!=lf2:
                    # open all switches
                    open_switch_all(dss,circuit, switch_names)
                    #disable_switches(dss,circuit, switch_names)
                    #dss.Solution.Solve()

                    # N-2
                    dss_node_voltage =  cause_line_fault(dss,str(lf))
                    dss_node_voltage2 =  cause_line_fault(dss,str(lf2))
                    

                    print('*****************************************')
                    print ('Contingency : Line Faults '+str(lf)+' and '+str(lf2))
                    # randomly search which switch to close or open 2^8 combination : 256 combinations
                    process_solution(dss,circuit,switch_names,critical_loads_bus,[lf,lf2])

                    # restore contingency
                    dss.run_command('Enable Line.'+str(lf))
                    dss.run_command('Enable Line.'+str(lf2))


    # fix an optimization problem based on minimization of curtailment of load.. solve the integer problem

    # define the objective and constraint for pyomo to solve

    # define the objective function

    # define the constraint


    # generate datasets for varying contingencies

    # discuss the contingencies to considered....store the sectionalizing switch status for those contingency