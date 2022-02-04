#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 LoRaSimMac v1.0: extension of LoRaSim
 									add simple carrier sense with backoff procedure and our proposed collision avoidance mechanism
 									can serve as a basis for simulating other MAC protocols for LoRa networks
 
 Copyright © 2021 Congduc Pham, University pf Pau	
 
 This work is licensed under the Creative Commons Attribution 4.0
 International License. To view a copy of this license,
 visit http://creativecommons.org/licenses/by/4.0/.
 
 Scientific references to cite if you are using this extension of the simulator:
 
 - for the early proposition on the collision avoidance mechanism

		"Dense Deployment of LoRa Networks: Expectations and Limits of Channel 
		Activity Detection and Capture Effect for Radio Channel Access"
		Congduc Pham and Muhammad Ehsan
		Sensors 2021, 21(3), 825; https://doi.org/10.3390/s21030825
		MDPI Sensors Journals. Massive and Reliable Sensor Communications with LPWANs Technologies special issue.
  
 List of additions to the original loraDir.py
 
 	 - add a uniform distribution [max(2000,node.period-5000),node.period+5000]
   - add support of LoRa 2.4GHz, sensitivity values for sf5->sf12 are taken from specs: lora24GHz = True
 	 - add sensitivity values for sf6 from specs
   - add simple carrier sense with backoff procedure: check_busy=True
   - add the proposed collision avoidance mechanism: CA=True
 
 the original loraDir.py which is pure ALOHA can be reproduced with
 	 - check_busy=False
   - CA=False	   
 	 
 this original loraDir.py can also benefit from a simple carrier sense with backoff procedure for comparison purposes:
   - enable carrier sense before transmission: check_busy=True
   - CCA reliability is controlled by CCA_prob: 100=always reliable to detect activity, 0=cannot detect activity, [1,99]in %
   - CCA_Prob=100 will give a DER of 1.0
   - with CA, you must also set n_retry_rts=-1 to get DER of 1.0
   		- this will put no limit on the number of retries when sending RTS
   - if channel busy, then can backoff by random backoff procedure controlled by Wbusy_min and Wbusy_BE
   		- [Wbusy_min, 2**Wbusy_BE]
   - backoff can be exponential with Wbusy_BE incremented at each retry: Wbusy_maxBE and Wbusy_exp_backoff=True
   - a maximum number of retries for data packet is implemented before packet transmission is aborted
   		- by default, we set it to 40 in order to avoid packet abortion
   		- this is useful when you want to know how many retries are currently performed
"""

"""
 LoRaSim 0.2.1: simulate collisions in LoRa
 Copyright © 2016 Thiemo Voigt <thiemo@sics.se> and Martin Bor <m.bor@lancaster.ac.uk>

 This work is licensed under the Creative Commons Attribution 4.0
 International License. To view a copy of this license,
 visit http://creativecommons.org/licenses/by/4.0/.

 Do LoRa Low-Power Wide-Area Networks Scale? Martin Bor, Utz Roedig, Thiemo Voigt
 and Juan Alonso, MSWiM '16, http://dx.doi.org/10.1145/2988287.2989163

 $Date: 2017-05-12 19:16:16 +0100 (Fri, 12 May 2017) $
 $Revision: 334 $
"""

"""
 SYNOPSIS:
   ./loraDir_mac.py <ca=1> <nodes> <avgsend> <experiment> <simtime> [collision] [WL] [W2] [W3] [Wnav] [W2afterNAV] [P] 
	 ./loraDir_mac.py <ca=0> <nodes> <avgsend> <experiment> <simtime> [collision]
 DESCRIPTION:
  ca
  	collision avoidance
  		set to 0 for the original behavior, i.e. ALOHA, equivalent to the original loraDir.py. carrier sense with backoff procedure can be enabled
  		set to 1 to enable the full collision avoidance mechanism, see scientific references for details
	nodes
		number of nodes to simulate
	avgsend
		average sending interval in milliseconds
	experiment
		experiment is an integer that determines with what radio settings the
		simulation is run. All nodes are configured with a fixed transmit power
		and a single transmit frequency, unless stated otherwise.
		0 use the settings with the the slowest datarate (SF12, BW125, CR4/8).
		1 similair to experiment 0, but use a random choice of 3 transmit
			frequencies.
		2 use the settings with the fastest data rate (SF6, BW500, CR4/5).
		3 optimise the setting per node based on the distance to the gateway.
		4 use the settings as defined in LoRaWAN (SF12, BW125, CR4/5).
		5 similair to experiment 3, but also optimises the transmit power.
		#C. Pham scenarios for Capture Effect in MDPI Sensors Journal
		#
		6 case A, 9 nodes, use the settings as defined in LoRaWAN (SF12, BW125, CR4/5)
		7 case B, 5 nodes, use the settings as defined in LoRaWAN (SF12, BW125, CR4/5)
	simtime
		total running time in milliseconds
	collision
		set to 1 to enable the full collision check, 0 to use a simplified check.
		With the simplified check, two messages collide when they arrive at the
		same time, on the same frequency and spreading factor. The full collision
		check considers the 'capture effect', whereby a collision of one or the
 OUTPUT
	The result of every simulation run will be appended to a file named expX.dat,
	whereby X is the experiment number. The file contains a space separated table
	of values for nodes, collisions, transmissions and total energy spent. The
	data file can be easily plotted using e.g. gnuplot.
"""

import simpy
import random
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import os

#proposed channel access mechanism
#WL=7 W2=10 W3=7 Wnav=0 W2afterNAV=7, carrier sense & exponential backoff can be enabled
#python loraDir_mac.py 1 20 20000 4 600000000 1 7 10 7 0 7
#-----------------------------------------------^
"""
BASIC USAGE EXAMPLE:

- we mainly use experiment 4: use the settings as defined in LoRaWAN (SF12, BW125, CR4/5) for all nodes
- define check_busy and CCA_prob according to desired behavior
- define targetSentPacket to the number of sent packet/node
- targetSentPacket*nrNodes will be the target total number of sent packets before we exit simulation
- change max_payload_size to the maximum packet length you want to consider, that impacts on the toa of the maximum packet length
- set distribType to either expoDistribType or uniformDistribType
- set packetLength to the packet length you want to simulate
- set print_sim = False to disable output on terminal to get faster execution
- then:

	> python loraDir_mac.py 1 20 20000 4 600000000 1 7 10 7 0 7

- simulation settings will be displayed

- at the end of the simulation, stats per nodes will be output in terminal with a summary of stats for the whole system

- DER is the Data Extraction Rate which is one of main the criteria to maximize
	- typical value with "python loraDir_mac.py 1 20 20000 4 600000000 1 7 10 7 0 7" -> DER: 0.336420884283

- summary of stats for the whole system will be appended to exp4.dat file

- re-run simulation by changing the packet inter-arrival time, smaller value will increase traffic load 

	> python loraDir_mac.py 1 20 20000 4 600000000 1 7 10 7 0 7
		+--------------------------^

ADVANCED USAGE EXAMPLE:

- change the common channel access parameters and the collision avoidance parameters under sections

#############################################
#common carrier sense & exponential backoff #
#############################################

##############
#only for CA #
##############

"""

#original loraDir.py with full_collision=1, carrier sense & exponential backoff can be enabled
"""
See above for BASIC USAGE EXAMPLE

- then:

	> python loraDir_mac.py 0 20 20000 4 600000000 1
	
- DER is the Data Extraction Rate which is one of main the criteria to maximize
	- typical value with "python loraDir_mac.py 0 20 20000 4 600000000 1" -> DER: 0.0421939523112

"""

#do the full collision check
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
full_collision = True

#check for channel busy or not, before phase 1 and before sending the data packet
check_busy = True

#turn on/off print
#disabling printing to stdout will make simulation much faster
print_sim = False
stdout_print_target=sys.stdout

#the selected packet length
packetLength=104

#SF value for experiment 4
exp4SF=12

#maximum of allowed payload size, have impact on NAV period
#LoRa Phy has a maximum of 255N but you can decide for smaller max value (as with SigFox for instance)
max_payload_size = 120

#nb of packet to be sent per node
#targetSentPacket*nrNodes will be the target total number of sent packets before we exit simulation
targetSentPacket=2000

#node type, you can add whatever you want to customize the behavior on configuration
endDeviceType=1
relayDeviceType=2

#packet transmit interval distribution type
expoDistribType=1
uniformDistribType=2

#the selected distribution
distribType=uniformDistribType
#distribType=expoDistribType

#keep the end simultion time, at which we reach targetSentPacket
endSim=0

#collision avoidance or not
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
CA = True
#CA1 variant, P=0 (all nodes start in phase 1) and no listening period in phase 2
CA1 = False
#CA2 variant, P=100 (all nodes start directly in phase 2) and W2 set to larger value, e.g. twice W2
CA2 = False

#set to True if LoRa 2.4Ghz is considered
lora24GHz = False

#turn on/off graphics
graphics = 0

#############################################
#common carrier sense & exponential backoff #
#############################################

#maximum number of retry when transmitting a data packet
n_retry=40
#maximum number of retry for RTS
n_retry_rts=20

#CCA reliability probability
#set to 0 to always assume that CCA indicates a free channel so that CA will be always used, or transmit immediately in ALOHA
#set to 100 for a fully reliable CCA, normally there should not be collision at all
#set to [1,99] to indicate a reliability percentage: i.e. (100-CCA_prob) is the probability that CCA reports a free channel while channel is busy
CCA_prob=50

#indicate whether channel is busy or not, we differentiate between channel_busy_rts and channel_busy_data
#to get more detailed statistics
channel_busy_rts = False
channel_busy_data = False

#minimun backoff when channel has been detected busy
Wbusy_min=1
#maximun backoff when channel has been detected busy 
Wbusy_BE=3
Wbusy_maxBE=6
#indicate whether the toa of the maximum allowed payload size should be added to backoff timer
#this is to avoid retrying during the packet transmission because of unreliable CCA
Wbusy_add_max_toa=False

#exponential backoff
Wbusy_exp_backoff=True

##############
#only for CA #
##############

#probability to start in phase 2
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
#we now mainly use P=0 when WL is small so the distinction between phase 1 and phase 2 only make sense when WL is large
P=0
#maximum number of DIFS for the listening period, IMPORTANT: changing here will not have any effect
#increase it if you want to extend the listening period duration 
WL=7
#for CA1 and CA2 variants to keep globally the same listening duration
CA1_WL = 2*WL
CA2_WL = 2*WL
# maximun backoff for phase 2
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
W2=10
initialW2=W2
# maximun backoff for phase 3
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
W3=7
# additional random backoff timer to a NAV period
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
Wnav=0
# maximum random backoff timer for phase 2 after a NAV period, set to W2 by default
#it is controlled from the command line, IMPORTANT: changing here will not have any effect
W2afterNAV=W2

#packet type
dataPacketType=1
rtsPacketType=2

#check for channel busy or not when sending RTS
check_busy_rts = True

#node's state to control the CA mechanism
schedule_tx=0
want_transmit=1
start_CA=2
start_phase1_listen=3
start_phase2_backoff=4
start_phase2_rts=5
start_phase2_listen=6
start_phase3_backoff=7
start_phase3_transmit=8
start_nav=9

#only for BW125, in nAh from SF7 to SF12
#based on SX1262 Semtech's AN on CAD performance
cad_consumption = [2.84, 5.75, 20.44, 41.36, 134.55, 169.54] 

##############
#end CA      #
##############

lastDisplayTime=-1

################################
# stats on inter-transmit time #
################################

n_transmit = 0
inter_transmit_time = 0
max_inter_transmit_time = 40
inter_transmit_time_bin=[]

for i in range(max_inter_transmit_time+1):
	inter_transmit_time_bin.append(0)	
			
last_transmit_time = 0

################################

# experiments:
# 0: packet with longest airtime, aloha-style experiment
# 0: one with 3 frequencies, 1 with 1 frequency
# 2: with shortest packets, still aloha-style
# 3: with shortest possible packets depending on distance

if lora24GHz:
	# this is an array with values for sensitivity
	# see SX128X Semtech doc
	# BW in 203, 406, 812, 1625 kHz
	sf5 = np.array([5,-109.0,-107.0,-105.0,-99.0])
	sf6 = np.array([6,-111.0,-110.0,-118.0,-103.0])
	sf7 = np.array([7,-115.0,-113.0,-112.0,-106.0])
	sf8 = np.array([8,-118.0,-116.0,-115.0,-109.0])
	sf9 = np.array([9,-121.0,-119.0,-117.0,-111.0])
	sf10 = np.array([10,-124.0,-122.0,-120.0,-114.0])
	sf11 = np.array([11,-127.0,-125.0,-123.0,-117.0])
	sf12 = np.array([12,-130.0,-128.0,-126.0,-120.0])
else:
	#taken for spec
	sf6 = np.array([6,-118.0,-115.0,-111.0])
	# this is an array with measured values for sensitivity
	# see paper, Table 3
	sf7 = np.array([7,-126.5,-124.25,-120.75])
	sf8 = np.array([8,-127.25,-126.75,-124.0])
	sf9 = np.array([9,-131.25,-128.25,-127.5])
	sf10 = np.array([10,-132.75,-130.25,-128.75])
	sf11 = np.array([11,-134.5,-132.75,-128.75])
	sf12 = np.array([12,-133.25,-132.25,-132.25])

#
# check for collisions at base station
# Note: called before a packet (or rather node) is inserted into the list
def checkcollision(packet):
	col = 0 # flag needed since there might be several collisions for packet
	processing = 0
	for i in range(0,len(packetsAtBS)):
		if packetsAtBS[i].packet.processed == 1 :
			processing = processing + 1
	if (processing > maxBSReceives):
		print "too long:", len(packetsAtBS)
		packet.processed = 0
	else:
		packet.processed = 1

	if packet.ptype == rtsPacketType:
		type_str="RTS"
	elif packet.ptype == dataPacketType:	
		type_str="DATA"
	else:
		type_str="N/A"
		
	print "*****> RCV at GW from node {} {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(packet.nodeid, type_str, packet.sf, packet.bw, packet.freq, len(packetsAtBS)) 

	global CA	
	
	if CA:
		print "- ",
		nbNodeListening=0

		for node in nodes:
			if node.ca_state==start_phase1_listen or node.ca_state==start_phase2_listen:
				if node.receive_rts==False and node.receive_data==False:
					nbNodeListening = nbNodeListening + 1
					print "{} - ".format(node.nodeid),

		print ""		
		print "There are {} nodes listening".format(nbNodeListening)
	
	if packetsAtBS:
		print "************************************************************************"
		print "CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(packet.nodeid, packet.sf, packet.bw, packet.freq, len(packetsAtBS))
		for other in packetsAtBS:
			if other.nodeid != packet.nodeid:
				if other.packet.ptype == rtsPacketType:
					type_str="RTS"
				elif other.packet.ptype == dataPacketType:	
					type_str="DATA"
				else:
					type_str="N/A"				
				print ">> node {} {} (sf:{} bw:{} freq:{:.6e})".format(other.nodeid, type_str, other.packet.sf, other.packet.bw, other.packet.freq)
				# simple collision
				if frequencyCollision(packet, other.packet) and sfCollision(packet, other.packet):
					if full_collision:
						if timingCollision(packet, other.packet):
							# check who collides in the power domain
							c = powerCollision(packet, other.packet)
							# mark all the collided packets
							# either this one, the other one, or both
							for p in c:
								p.collided = 1
								if p == packet:
									col = 1
						else:							 
							# no timing collision, all fine
							pass
					else:
						packet.collided = 1
						other.packet.collided = 1	 # other also got lost, if it wasn't lost already
						col = 1			

		print "Summary: ",
		print "Packet from {}(".format(packet.nodeid),
		if packet.collided:
			print "collided) ",
		else:
			print "ok) ",
		for other in packetsAtBS:
			print "Packet from {}(".format(other.nodeid),
			if other.packet.collided:
				print "collided) ",
			else:
				print "ok) ",			 		

		print ""
		
		if CA:
			#we have to correct previous decision as previous RTS or DATA packets can be now marked as collided
			#their state can still be listening, we just cancel the fact that they received an RTS or DATA
			for other in packetsAtBS:
				for node in nodes:
					if node.receive_rts==True and node.receive_rts_from==other.nodeid and other.packet.collided:
						print "** node {} cancel reception of RTS from node {} due to collision".format(node.nodeid, other.nodeid)
						node.receive_rts=False
					if node.receive_data==True and node.receive_data_from==other.nodeid and other.packet.collided:
						print "** node {} cancel reception of ValidHeader from node {} due to collision".format(node.nodeid, other.nodeid)					
						node.receive_data=False
		
	print "========================================================================"	
	
	#if col==1 it means that the new packet can not be decoded
	if col:
		return col	
			
	#normally, here, the packet has been correctly received	
	print "GW got packet from node {}".format(packet.nodeid)

	if CA:
		#the trick is to assume that if the gateway received a packet
		#then all other nodes in the listening period should also have receive it
		#there might be some cases where a geographically central gw would have received a packet while a distant node,
		#far from the transmitter node might not receive the packet. But here we assume that the distances allow such reception
		for node in nodes:
			if node.nodeid != packet.nodeid:
				if node.ca_state==start_phase1_listen or node.ca_state==start_phase2_listen:
					#either we receive an RTS, either it is a DATA
					#once we receive RTS or DATA we normally leave listen state to go into NAV
					if node.receive_rts==False and node.receive_data==False:
						if packet.ptype==rtsPacketType:
							node.receive_rts=True
							node.receive_rts_from=packet.nodeid
							print "-- node {} marked to have received RTS from node {}".format(node.nodeid, packet.nodeid)
							#keep track of when the RTS should have been received
							node.receive_rts_time=env.now
							#for an RTS packet, packet.data_len stores the data packet length
							#set node.nav to the size of the forthcoming data packet
							node.nav=packet.data_len
						if packet.ptype==dataPacketType:
							node.receive_data=True
							node.receive_data_from=packet.nodeid
							print "-- node {} marked to have received ValidHeader from node {}".format(node.nodeid, packet.nodeid)					
							#keep track of when the DATA should have been received
							node.receive_data_time=env.now
							#for an DATA packet we take the maximum length
							node.nav=max_payload_size						
	print "========================================================================"
	return 0

#
# frequencyCollision, conditions
#
#		 |f1-f2| <= 120 kHz if f1 or f2 has bw 500
#		 |f1-f2| <= 60 kHz if f1 or f2 has bw 250
#		 |f1-f2| <= 30 kHz if f1 or f2 has bw 125
def frequencyCollision(p1,p2):
	if (abs(p1.freq-p2.freq)<=120 and (p1.bw==500 or p2.freq==500)):
		print "frequency coll 500"
		return True
	elif (abs(p1.freq-p2.freq)<=60 and (p1.bw==250 or p2.freq==250)):
		print "frequency coll 250"
		return True
	else:
		if (abs(p1.freq-p2.freq)<=30):
			print "frequency coll 125"
			return True
		#else:
	print "no frequency coll"
	return False

def sfCollision(p1, p2):
	if p1.sf == p2.sf:
		print "collision sf node {} and node {}".format(p1.nodeid, p2.nodeid)
		# p2 may have been lost too, will be marked by other checks
		return True
	print "no sf collision"
	return False

def powerCollision(p1, p2):
	powerThreshold = 6 # dB
	print "pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2))
	if abs(p1.rssi - p2.rssi) < powerThreshold:
		print "collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid)
		# packets are too close to each other, both collide
		# return both packets as casualties
		return (p1, p2)
	elif p1.rssi - p2.rssi < powerThreshold:
		# p2 overpowered p1, return p1 as casualty
		print "collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid)
		return (p1,)
	print "p1 wins, p2 lost"
	# p2 was the weaker packet, return it as a casualty
	return (p2,)

def timingCollision(p1, p2):
	# assuming p1 is the freshly arrived packet and this is the last check
	# we've already determined that p1 is a weak packet, so the only
	# way we can win is by being late enough (only the first n - 5 preamble symbols overlap)

	# assuming 8 preamble symbols
	Npream = 8

	# we can lose at most (Npream - 5) * Tsym of our preamble
	Tpreamb = 2**p1.sf/(1.0*p1.bw) * (Npream - 5)

	# check whether p2 ends in p1's critical section
	p2_end = p2.addTime + p2.rectime
	p1_cs = env.now + Tpreamb
	print "collision timing node {} ({},{},{}) node {} ({},{})".format(
		p1.nodeid, env.now - env.now, p1_cs - env.now, p1.rectime,
		p2.nodeid, p2.addTime - env.now, p2_end - env.now
	)
	if p1_cs < p2_end:
		# p1 collided with p2 and lost
		print "not late enough"
		return True
	print "saved by the preamble"
	return False

# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf
#
def airtime(sf,cr,pl,bw):
	
	DE = 0		 # low data rate optimization enabled (=1) or not (=0)
	Npream = 8	 # number of preamble symbol (12.25	 from Utz paper)

	if lora24GHz:
		Npream = 12
		H = 1		 # header for variable length packet (H=1) or not (H=0)		
		if sf > 10:
			# low data rate optimization mandated for SF > 10
			DE = 1
		Tsym = (2.0**sf)/bw
		if sf < 7:
			Tpream = (Npream + 6.25)*Tsym
		else:
			Tpream = (Npream + 4.25)*Tsym
		#print "sf", sf, " cr", cr, "pl", pl, "bw", bw
		if sf >= 7:
			payloadSymbNB = 8 + math.ceil(max((8.0*pl+16-4.0*sf+8+20*H),0)/(4.0*(sf-2*DE)))*(cr+4)
		else:
			payloadSymbNB = 8 + math.ceil(max((8.0*pl+16-4.0*sf+20*H),0)/(4.0*(sf-2*DE)))*(cr+4)
		Tpayload = payloadSymbNB * Tsym
		return Tpream + Tpayload		
	else:
		H = 0		 # implicit header disabled (H=0) or not (H=1)
		if bw == 125 and sf in [11, 12]:
			# low data rate optimization mandated for BW125 with SF11 and SF12
			DE = 1
		if sf == 6:
			# can only have implicit header with SF6
			H = 1
		Tsym = (2.0**sf)/bw
		Tpream = (Npream + 4.25)*Tsym
		#print "sf", sf, " cr", cr, "pl", pl, "bw", bw
		payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
		Tpayload = payloadSymbNB * Tsym
		return Tpream + Tpayload
	
#
# this function creates a node
#
class myNode():
	def __init__(self, nodeid, nodeType, bs, period, distrib, packetlen):
		self.nodeid = nodeid
		self.nodeType = nodeType		
		self.period = period
		self.distrib = distrib
		self.bs = bs
		self.x = 0
		self.y = 0

		global expoDistribType
		global uniformDistribType
		global endDeviceType
		global relayDeviceType
		
		# this is very complex prodecure for placing nodes
		# and ensure minimum distance between each pair of nodes
		found = 0
		rounds = 0
		global nodes
		while (found == 0 and rounds < 100):
			a = random.random()
			b = random.random()
			if b<a:
				a,b = b,a
			posx = b*maxDist*math.cos(2*math.pi*a/b)+bsx
			posy = b*maxDist*math.sin(2*math.pi*a/b)+bsy
			if len(nodes) > 0:
				for index, n in enumerate(nodes):
					dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2))
					if dist >= 10:
						found = 1
						self.x = posx
						self.y = posy
					else:
						rounds = rounds + 1
						if rounds == 100:
							print "could not place new node, giving up"
							exit(-1)
			else:
				print "first node"
				self.x = posx
				self.y = posy
				found = 1
		self.dist = np.sqrt((self.x-bsx)*(self.x-bsx)+(self.y-bsy)*(self.y-bsy))
		print('node %d %s %s' % (nodeid,  'endDevice' if self.nodeType==endDeviceType else 'relayDevice', \
																			'expo' if self.distrib==expoDistribType else 'uniform'), \
														"x", self.x, "y", self.y, "dist: ", self.dist)
		
		self.packet = myPacket(self.nodeid, packetlen, self.dist)
		self.data_len=packetlen
		
		self.data_rectime = airtime(self.packet.sf,self.packet.cr,self.packet.pl,self.packet.bw)
		print "rectime for DATA packet ", self.data_rectime
		self.rts_rectime = airtime(self.packet.sf,self.packet.cr,5,self.packet.bw)
		print "rectime for RTS packet ", self.rts_rectime
		self.n_data_sent = 0
		self.n_rts_sent = 0
		self.ca_state = schedule_tx
		self.want_transmit_time=0
		self.ca_listen_start_time = 0
		self.ca_listen_end_time = 0
		self.total_listen_time = 0
		global P
		self.P=P
		self.my_P=0
		self.backoff=0
		self.receive_rts=False
		self.receive_rts_time=0
		self.receive_rts_from=-1
		self.n_receive_nav_rts_p1=0
		self.n_receive_nav_rts_p2=0
		self.receive_data=False
		self.receive_data_time=0
		self.receive_data_from=-1
		self.n_receive_nav_data_p1=0
		self.n_receive_nav_data_p2=0		
		self.nav=0
		self.cca=False
		self.n_cca=0
		self.n_busy_rts=0
		self.n_busy_rts_p1=0		
		self.n_busy_data=0
		global n_retry
		self.n_retry=n_retry
		self.total_retry=0
		self.retry_bin=[]
		for i in range(n_retry):
			self.retry_bin.append(0)		
		global n_retry_rts
		if n_retry_rts>0:
			self.n_retry_rts=n_retry_rts
		else:
			#if n_retry_rts<0 then we will not decrement node.n_retry_rts
			self.n_retry_rts=1
		self.total_retry_rts=0
		self.retry_rts_bin=[]
		for i in range(n_retry_rts+1):
			self.retry_rts_bin.append(0)						
		self.n_aborted=0
		self.cycle=0
		global W2
		self.W2=W2
		self.latency=0
		global Wbusy_BE
		self.Wbusy_BE=Wbusy_BE
		
		# graphics for node
		global graphics
		if (graphics == 1):
			global ax
			ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='blue'))

#
# this function creates a packet (associated with a node)
# it also sets all parameters, currently random
#
class myPacket():
	def __init__(self, nodeid, plen, distance):
		global experiment
		global Ptx
		global gamma
		global d0
		global var
		global Lpld0
		global GL
		
		global exp4SF

		self.nodeid = nodeid
		self.txpow = Ptx

		# randomize configuration values
		if lora24GHz:
			self.sf = random.randint(5,12)
			self.bw = random.choice([203.125, 406.250, 812.5, 1625])
		else:	
			self.sf = random.randint(6,12)
			self.bw = random.choice([125, 250, 500])
		self.cr = random.randint(1,4)

		# for certain experiments override these
		if experiment==1 or experiment == 0:
			self.sf = 12
			self.cr = 4
			if lora24GHz:
				self.bw = 203.125
			else:	
				self.bw = 125

		# for certain experiments override these
		if experiment==2:
			if lora24GHz:
				self.sf = 5
				self.cr = 1
				self.bw = 1625			
			else:
				self.sf = 6
				self.cr = 1
				self.bw = 500
		# lorawan
		if experiment in [4,6,7]:
			if lora24GHz:
				self.bw = 203.125			
			else:
				self.bw = 125
			self.sf = exp4SF
			self.cr = 1				

		# for experiment 3 find the best setting
		# OBS, some hardcoded values
		Prx = self.txpow	## zero path loss by default

		# log-shadow
		Lpl = Lpld0 + 10*gamma*math.log10(distance/d0)
		print "Lpl:", Lpl
		Prx = self.txpow - GL - Lpl

		#TODO for lora24GHz
		if (experiment == 3) or (experiment == 5):
			minairtime = 9999
			minsf = 0
			minbw = 0

			print "Prx:", Prx

			for i in range(0,6):
				for j in range(1,4):
					if (sensi[i,j] < Prx):
						self.sf = int(sensi[i,0])
						if j==1:
							self.bw = 125
						elif j==2:
							self.bw = 250
						else:
							self.bw=500
						at = airtime(self.sf, 1, plen, self.bw)
						if at < minairtime:
							minairtime = at
							minsf = self.sf
							minbw = self.bw
							minsensi = sensi[i, j]
			if (minairtime == 9999):
				print "does not reach base station"
				exit(-1)
			print "best sf:", minsf, " best bw: ", minbw, "best airtime:", minairtime
			self.rectime = minairtime
			self.sf = minsf
			self.bw = minbw
			self.cr = 1

			if experiment == 5:
				# reduce the txpower if there's room left
				self.txpow = max(2, self.txpow - math.floor(Prx - minsensi))
				Prx = self.txpow - GL - Lpl
				print 'minsesi {} best txpow {}'.format(minsensi, self.txpow)

		# transmission range, needs update XXX
		self.transRange = 150
		self.pl = plen
		self.symTime = (2.0**self.sf)/self.bw
		self.arriveTime = 0
		self.rssi = Prx
		# frequencies: lower bound + number of 61 Hz steps
		if lora24GHz:
			self.freq = 2403000000 + random.randint(0,2622950)
		else:
			self.freq = 860000000 + random.randint(0,2622950)

		# for certain experiments override these and
		# choose some random frequences
		if experiment == 1:
			if lora24GHz:
				self.freq = random.choice([2403000000, 2425000000, 2479000000])
			else:
				self.freq = random.choice([860000000, 864000000, 868000000])
		else:
			if lora24GHz:
				self.freq = 2403000000
			else:
				self.freq = 860000000
	
		print "frequency" ,self.freq, "symTime ", self.symTime
		print "bw", self.bw, "sf", self.sf, "cr", self.cr, "rssi", self.rssi

		self.ptype = dataPacketType
		#self.data_len will keep the payload length of a data packet
		#self.pl will be used to keep the current packet length which can either be 
		#data_len or 5 (the size of an RTS packet)
		self.data_len=plen
		if lora24GHz:
			Npream = 12	 
			if self.sf < 7:
				self.Tpream = (Npream + 6.25)*self.symTime
			else:
				self.Tpream = (Npream + 4.25)*self.symTime		 		
		else:		
			Npream = 8	 # number of preamble symbol (12.25	 from Utz paper) 
			self.Tpream = (Npream + 4.25)*self.symTime		
		self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)
		print "rectime node ", self.nodeid, "	 ", self.rectime
		print "T_Pream node ", self.nodeid, "	 ", self.Tpream		
		# denote if packet is collided
		self.collided = 0
		self.processed = 0

	def setPacketType(self,ptype):
		self.ptype = ptype
		
		if ptype == rtsPacketType:
			self.pl=5
			self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)		 
		else:
			self.pl=self.data_len
			self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)
		

#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#
"""
the full collision avoidance mechanism is implemented here where the node can enter into several states:
schedule_tx, want_transmit, start_CA, start_phase1_listen, start_phase2_backoff, start_phase2_rts, start_phase2_listen,
start_phase3_backoff, start_phase3_transmit and start_nav

after schedule_tx, then collision avoidance works in 3 phases: 
	- phase1 (want_transmit, start_CA, start_phase1_listen)
	- phase2 (start_phase2_backoff, start_phase2_rts, start_phase2_listen)
	- phase3 (start_phase3_backoff, start_phase3_transmit)

CCA can be realized (check_busy==True) at want_transmit (DATA), start_phase2_rts (RTS) and start_phase3_transmit (DATA)

- node.n_retry controls how many retries is allowed at want_transmit for DATA
- node.n_retry_rts controls how many retries is allowed at start_phase2_rts for RTS
- Exponential backoff [Wbusy_min,2**node.Wbusy_BE] is used for both procedures
	- if Wbusy_exp_backoff==True then node.Wbusy_BE is incremented at each retry if node.Wbusy_BE < Wbusy_maxBE
- For DATA,
	- at want_transmit, after node.n_retry, packet transmission is aborted and node.latency not updated
	- at start_phase3_transmit, if channel is busy then we go back to want_transmit, node.n_retry is decremented and packet TX can then be aborted in want_transmit
- For RTS, at start_phase2_rts, after node.n_retry_rts, the RTS is transmitted even if channel is busy
- A node will enter into NAV period upon reception of an RTS or a ValidHeader from DATA
	- the node will go back to want_transmit, node.n_retry is decremented and packet TX can then be aborted in want_transmit
- If eventually the data packet is transmitted, then node.latency is updated 
"""

def transmit(env,node):
	while True:
		global nrLost
		global nrCollisions
		global nrReceived
		global nrProcessed

		global WL
		global W2
		global W3	
		global Wnav			
		global W2afterNAV
		global CA
		global CA1
		global CA2
		global experiment
		global check_busy
		global check_busy_rts
		global CCA_prob
		global channel_busy_rts
		global channel_busy_data
		global Wbusy_min
		global Wbusy_BE
		global Wbusy_maxBE
		global Wbusy_exp_backoff
		global n_retry
		global n_retry_rts
		
		global expoDistribType
		global uniformDistribType
		global endDeviceType
		global relayDeviceType
		
 		global n_transmit
 		global inter_transmit_time
 		global inter_transmit_time_bin
		global last_transmit_time
		
		print "node {}: transmit() simTime {}".format(node.nodeid, env.now)
		
		###////////////////////////////////////////////////////////
		# Collision Avoidance                                     /
		###////////////////////////////////////////////////////////		
		if CA:
			###############################
			# schedule_tx                 #
			###############################
			#if node.nav==0, then it is a new transmission attempt		
			if node.ca_state==schedule_tx:
				if experiment==6:
					#normally 9 nodes with 100ms delay between each node
					transmit_wait=node.cycle*node.period-env.now+node.nodeid*100
				elif experiment==7:
					#normally 5 nodes with 500ms delay between each node
					transmit_wait=node.cycle*node.period-env.now+node.nodeid*500			
				else:
					if node.distrib==expoDistribType:
						transmit_wait = random.expovariate(1.0/float(node.period))
					if node.distrib==uniformDistribType:
						transmit_wait = random.uniform(max(2000,node.period-5000),node.period+5000)		
				
				print "node {} {} cycle {}: schedule transmit in {} at {}".format(node.nodeid, env.now, node.cycle, transmit_wait, env.now+transmit_wait)

				node.cycle = node.cycle + 1
				
				node.ca_state=want_transmit
				
				yield env.timeout(transmit_wait)

			###############################
			# want_transmit -> start_CA   #
			###############################	
			if node.ca_state==want_transmit and node.packet.ptype==dataPacketType:
				if node.n_retry==0:
					print "node {} {}: current transmission aborted".format(node.nodeid, env.now)
					node.n_aborted = node.n_aborted +1
					#reset for sending a new packet				
					node.n_retry=n_retry
					node.Wbusy_BE=Wbusy_BE
					node.cca=False
					node.nav=0
					#node will then try immediately to send new packet					
				else:					
					if node.cca:
						#reset cca to start again
						node.cca=False
						print "node {} {}: retry {} after CCA".format(node.nodeid, env.now, n_retry-node.n_retry)									
					elif node.nav!=0:
						#reset nav to start again a complete CA procedure
						node.nav=0						
						#will we use W2afterNAV after a NAV period?
						if W2afterNAV!=W2:
							node.W2=W2afterNAV
							print "node {} {}: retry {} after NAV -> W2=W2afterNAV={}".format(node.nodeid, env.now, n_retry-node.n_retry, node.W2)
							#TODO still need to see where we are going to introduce W2afterNAV
						else:
							node.W2=W2
							print "node {} {}: retry {} after NAV -> W2={}".format(node.nodeid, env.now, n_retry-node.n_retry, node.W2)						
					else:
						#this is an initial transmit attempt
						node.want_transmit_time=env.now
						n_transmit = n_transmit + 1
						if n_transmit > 1:
							current_inter_transmit_time = env.now - last_transmit_time
							inter_transmit_time += current_inter_transmit_time
							#put in bin from 0s to max_inter_transmit_time in second
							inter_transmit_time_bin[min(int(current_inter_transmit_time/1000), max_inter_transmit_time)] += 1	
						last_transmit_time = env.now		
						
					channel_find_busy=False
				
					if check_busy:
						node.n_cca = node.n_cca + 1
						print "node {} {}: CA want_transmit checking channel".format(node.nodeid, env.now)
						#if channel is busy, then CCA reliability will decide if we can detect that channel is busy
						#if channel is not busy, as we observed no false positive, so channel_find_busy remains False
						if channel_busy_rts or channel_busy_data:
							print "node {}: channel is busy by {}".format(node.nodeid, 'RTS' if channel_busy_rts else 'DATA')
							if channel_busy_rts:
								node.n_busy_rts += 1
								node.n_busy_rts_p1 += 1
							else:
								node.n_busy_data += 1	
							if random.randint(1,100) <= CCA_prob and CCA_prob!=0:
								channel_find_busy=True
								print "node {}: channel found busy by CCA with {}%".format(node.nodeid, CCA_prob)
							else:
								channel_find_busy=False
								print "node {}: channel found free by CCA".format(node.nodeid)		
						else:
							print "node {}: channel is free".format(node.nodeid)						
				
					#print "node {}: TEST -> force channel found free".format(node.nodeid)
					#channel_find_busy=False
				
					if channel_find_busy:
						#here we just delay by a random backoff timer to retry again
						#random backoff [Wbusy_min,2**Wbusy_BE]
						node.backoff=random.randint(Wbusy_min,2**node.Wbusy_BE)
						print "node {}: channel found busy, backoff with Wbusy=[{},{}] backoff={} DIFS={}".format(node.nodeid, Wbusy_min, 2**node.Wbusy_BE, node.backoff, node.packet.Tpream)
						node.cca=True
						if Wbusy_exp_backoff:
							if node.Wbusy_BE<Wbusy_maxBE:
								node.Wbusy_BE=node.Wbusy_BE + 1
						print "node {}: number of retries left {}".format(node.nodeid, node.n_retry)		
						node.n_retry = node.n_retry - 1							
						yield env.timeout(node.backoff*node.packet.Tpream)
					else:					
						#determine if the node starts in phase 1 (listen for RTS) or in phase 2 (send RTS after backoff)
						node.my_P=random.randint(0,100)
						node.ca_state=start_CA
						print "node {} {}: start_CA with P={} my_P={}".format(node.nodeid, env.now, node.P, node.my_P)
						#change packet type to get the correct time-on-air
						node.packet.setPacketType(rtsPacketType)

			#########################################################
			# start_CA -> start_phase_listen | start_phase2_backoff #
			#########################################################
			if node.ca_state==start_CA and node.packet.ptype==rtsPacketType:							
				if node.my_P > node.P:	
					#starts in phase 1
					node.ca_state=start_phase1_listen
					#store time at which listening period began
					node.ca_listen_start_time=env.now
					node.ca_listen_end_time=env.now+(WL*node.packet.Tpream+node.packet.rectime)
					print "node {} {}: start_phase1_listen with WL={} DIFS={} TOA(RTS)={} until {}".format(node.nodeid, env.now, WL, node.packet.Tpream, node.packet.rectime, node.ca_listen_end_time)					
					#listen period is at least WL*DIFS+TOA(RTS), with DIFS=preamble duration
					yield env.timeout(WL*node.packet.Tpream+node.packet.rectime)
				else:
					#starts in phase 2
					node.ca_state=start_phase2_backoff
					if CA2:
						#here, we decided to keep same W2, but your change it for CA2 specifically
						#for instance random backoff [0,2*W2]
						node.backoff=random.randint(0,W2)
						print "node {} {}: CA2 variant".format(node.nodeid, env.now)
						print "node {} {}: start_phase2_backoff with CA2_W2={} backoff={} DIFS={}".format(node.nodeid, env.now, W2, node.backoff, node.packet.Tpream)					
					else:
						#random backoff [0,W2]
						node.backoff=random.randint(0,W2)
						print "node {} {}: start_phase2_backoff with W2={} backoff={} DIFS={}".format(node.nodeid, env.now, W2, node.backoff, node.packet.Tpream)
					#backoff period is backoff*DIFS, with DIFS=preamble duration
					yield env.timeout(node.backoff*node.packet.Tpream)

			###########################################################
			# start_phase1_listen -> start_nav | start_phase2_backoff #
			###########################################################
			#node was in start_phase1_listen and it did not receive an RTS
			if node.ca_state==start_phase1_listen and node.receive_rts==False:
				#did we receive a DATA with a ValidHeader?
				if node.receive_data==True:
					node.total_listen_time = node.total_listen_time + (node.receive_data_time - node.ca_listen_start_time) 
					node.receive_data = False
					node.n_receive_nav_data_p1 = node.n_receive_nav_data_p1 + 1
					#nav period is the time-on-air of the maximum data size which is returned in node.nav
					nav_period=airtime(node.packet.sf,node.packet.cr,node.nav,node.packet.bw)
					#will go into NAV
					node.ca_state=start_nav				
					#add an additional number of random DIFS [0,Wnav]
					if Wnav!=0:
						extra_nav_difs=random.randint(0,Wnav)
					else:
						extra_nav_difs=0									
					#it can happen that the end of the listening period is after the theoretical NAV period	for data packet
					#in this case, it is not really possible to revert time and the end of the listening period will be the end of the nav period
					if node.receive_data_time+nav_period+extra_nav_difs*node.packet.Tpream <= env.now:
						#in this case, there is no additional delay, we just go to start_nav
						print "node {} {}: received ValidHeader at {}, NAV period is included in listening period".format(node.nodeid, env.now, node.receive_data_time)
					else:						
						print "node {} {}: received ValidHeader at {} go into NAV({}) + [0,{}]{} DIFS until {}".format(node.nodeid, env.now, node.receive_data_time, nav_period, Wnav, extra_nav_difs, node.receive_data_time+nav_period+extra_nav_difs*node.packet.Tpream)
						#adjust to remove the extra time due to the fact that the data should have been received ealier					
						nav_period=nav_period+extra_nav_difs*node.packet.Tpream-(env.now-node.receive_data_time)					
						yield env.timeout(nav_period)			
				else:
					#random backoff [0,W2]
					node.backoff=random.randint(0,W2)
					print "node {} {}: start_phase2_backoff with W2={} backoff={} DIFS={}".format(node.nodeid, env.now, W2, node.backoff, node.packet.Tpream)				
					#starts phase 2
					node.ca_state=start_phase2_backoff
					#backoff period is backoff*DIFS, with DIFS=preamble duration
					yield env.timeout(node.backoff*node.packet.Tpream)		

			###########################################################
			# start_phase1_listen -> start_nav(RTS)                   #
			###########################################################			
			#node was in start_phase1_listen and it did receive an RTS
			#we process this event at the end of the listening period, normally the RTS has been received in the past
			if node.ca_state==start_phase1_listen and node.receive_rts==True:
				node.receive_rts = False
				node.total_listen_time = node.total_listen_time + (node.receive_rts_time - node.ca_listen_start_time)
				node.n_receive_nav_rts_p1 = node.n_receive_nav_rts_p1 + 1
				#nav period is one listening period + W3*DIFS + TOA(data)
				nav_period=WL*node.packet.Tpream+node.packet.rectime + W3*node.packet.Tpream + airtime(node.packet.sf,node.packet.cr,node.nav,node.packet.bw)
				#add an additional number of random DIFS [0,Wnav]
				if Wnav!=0:
					extra_nav_difs=random.randint(0,Wnav)
				else:
					extra_nav_difs=0
				print "node {} {}: received RTS at {} go into NAV({}) + [0,{}]{} DIFS until {}".format(node.nodeid, env.now, node.receive_rts_time, nav_period, Wnav, extra_nav_difs, node.receive_rts_time+nav_period+extra_nav_difs*node.packet.Tpream)
				#adjust to remove the extra time due to the fact that the RTS should have been received ealier
				nav_period=nav_period+extra_nav_difs*node.packet.Tpream-(env.now-node.receive_rts_time)
				#go into NAV
				node.ca_state=start_nav			
				yield env.timeout(nav_period)

			###########################################################
			# start_phase2_backoff -> start_phase2_rts                #
			###########################################################				
			if node.ca_state==start_phase2_backoff:		
				#we sent the RTS
				node.ca_state=start_phase2_rts
				#reset node.Wbusy_BE for the RTS
				node.Wbusy_BE=Wbusy_BE
				#if n_retry_rts<0 then we will not decrement node.n_retry_rts
				#so no need to re-initialize it
				if n_retry_rts>0:
					node.n_retry_rts=n_retry_rts				
				node.cca=False
								
				channel_find_busy=True
			
				while node.n_retry_rts and channel_find_busy:
					if check_busy_rts:
						node.n_cca = node.n_cca + 1
						print "node {} {}: phase2 prior to send RTS checking channel".format(node.nodeid, env.now)
						#if channel is busy, then CCA reliability will decide if we can detect that channel is busy
						#if channel is not busy, as we observed no false positive, so channel_find_busy remains False
						if channel_busy_rts or channel_busy_data:
							print "node {}: channel is busy by {}".format(node.nodeid, 'RTS' if channel_busy_rts else 'DATA')
							if channel_busy_rts:
								node.n_busy_rts += 1
							else:
								node.n_busy_data += 1
							if random.randint(1,100) <= CCA_prob and CCA_prob!=0:
								channel_find_busy=True
								print "node {}: channel found busy by CCA with {}%".format(node.nodeid, CCA_prob)
							else:
								channel_find_busy=False
								print "node {}: channel found free by CCA".format(node.nodeid)		
						else:
							channel_find_busy=False
							print "node {}: channel is free".format(node.nodeid)						
					else:
						channel_find_busy=False
				
					#print "node {}: TEST -> force channel found free".format(node.nodeid)
					#channel_find_busy=False		
			
					if channel_find_busy:
						#here we just delay by a random backoff timer to retry again
						#random backoff [Wbusy_min,2**Wbusy_BE]
						node.backoff=random.randint(Wbusy_min,2**node.Wbusy_BE)
						print "node {}: channel found busy, backoff with Wbusy=[{},{}] backoff={} DIFS={}".format(node.nodeid, Wbusy_min, 2**node.Wbusy_BE, node.backoff, node.packet.Tpream)
						if Wbusy_exp_backoff:
							if node.Wbusy_BE<Wbusy_maxBE:
								node.Wbusy_BE=node.Wbusy_BE + 1
						print "node {}: number of retries left {}".format(node.nodeid, node.n_retry_rts)
						#if n_retry_rts<0 then we will not decrement node.n_retry_rts
						if n_retry_rts>0:		
							node.n_retry_rts = node.n_retry_rts - 1
						yield env.timeout(node.backoff*node.packet.Tpream)

				#after n_retry_rts, we transmit anyway
				if node.n_retry_rts==0:
					print "node {}: {} RTS max number of transmission reached, transmit anyway".format(node.nodeid, n_retry_rts)

				# RTS time sending and receiving
				# RTS packet arrives -> add to base station
				print "node {} {}: transmit RTS toa {} transmission ends at {}".format(node.nodeid, env.now, node.packet.rectime, env.now+node.packet.rectime)
				node.n_rts_sent = node.n_rts_sent + 1
				node.total_retry_rts += n_retry_rts - node.n_retry_rts
				node.retry_rts_bin[n_retry_rts - node.n_retry_rts] += 1				
				if (node in packetsAtBS):
					print "ERROR: RTS packet already in"
				else:
					if lora24GHz:
						sensitivity = sensi[node.packet.sf - 5, [203.125,406.25,812.5,1625].index(node.packet.bw) + 1]
					else:
						sensitivity = sensi[node.packet.sf - 6, [125,250,500].index(node.packet.bw) + 1]
					if node.packet.rssi < sensitivity:
						print "node {} {}: RTS packet will be lost".format(node.nodeid, env.now)
						node.packet.lost = True
					else:
						node.packet.lost = False
						checkcollision(node.packet)
						packetsAtBS.append(node)
						node.packet.addTime = env.now

				channel_busy_rts=True
				yield env.timeout(node.packet.rectime)
				channel_busy_rts=False
				
				if node.packet.lost:
					global nrRTSLost
					nrRTSLost += 1
				if node.packet.collided == 1:
					global nrRTSCollisions
					nrRTSCollisions = nrRTSCollisions +1
				if node.packet.collided == 0 and not node.packet.lost:
					global nrRTSReceived
					nrRTSReceived = nrRTSReceived + 1
					print "node {} {}: RTS packet has been correctly transmitted".format(node.nodeid, env.now)
				if node.packet.processed == 1:
					global nrRTSProcessed
					nrRTSProcessed = nrRTSProcessed + 1

				# complete packet has been received by base station
				# can remove it
				if (node in packetsAtBS):
					packetsAtBS.remove(node)
				# reset the packet
				node.packet.collided = 0
				node.packet.processed = 0
				node.packet.lost = False			

			###########################################################
			# start_phase2_rts -> start_phase2_listen                 #
			###########################################################				
			if node.ca_state==start_phase2_rts:
				if CA1:
					#CA1 variant removes listening in phase 2
					#starts phase 3
					node.ca_state=start_phase3_backoff
					#random backoff [0,W3]
					node.backoff=random.randint(0,W3)
					print "node {} {}: CA1 variant".format(node.nodeid, env.now)
					print "node {} {}: start_phase3_backoff with W3={} backoff={} DIFS={}".format(node.nodeid, env.now, W3, node.backoff, node.packet.Tpream)
					#backoff period is backoff*DIFS, with DIFS=preamble duration
					yield env.timeout(node.backoff*node.packet.Tpream)
				else:					
					#we have sent RTS, so go for another listening period
					node.ca_state=start_phase2_listen			
					#store time at which listening period began
					node.ca_listen_start_time=env.now
					node.ca_listen_end_time=env.now+(WL*node.packet.Tpream+node.packet.rectime)
					print "node {} {}: start_phase2_listen with WL={} DIFS={} TOA(RTS)={} until {}".format(node.nodeid, env.now, WL, node.packet.Tpream, node.packet.rectime, node.ca_listen_end_time)
					#listen period is at least WL*DIFS+TOA(RTS), with DIFS=preamble duration
					yield env.timeout(WL*node.packet.Tpream+node.packet.rectime)

			###########################################################
			# start_phase2_listen -> start_nav | start_phase3_backoff #
			###########################################################
			#node was in start_phase2_listen and it did not receive an RTS
			if node.ca_state==start_phase2_listen and node.receive_rts==False:
				#did we receive a DATA with a ValidHeader?
				if node.receive_data==True:
					node.receive_data = False
					node.total_listen_time = node.total_listen_time + (node.receive_data_time - node.ca_listen_start_time)
					node.n_receive_nav_data_p2 = node.n_receive_nav_data_p2 + 1				
					#nav period is the time-on-air of the maximum data size which is returned in node.nav
					nav_period=airtime(node.packet.sf,node.packet.cr,node.nav,node.packet.bw)
					#will go into NAV
					node.ca_state=start_nav
					#add an additional number of random DIFS [0,Wnav]
					if Wnav!=0:
						extra_nav_difs=random.randint(0,Wnav)
					else:
						extra_nav_difs=0												
					#it can happen that the end of the listening period is after the theoretical NAV period for data packet
					#in this case, it is not really possible to revert time and the end of the listening period will be the end of the nav period
					if node.receive_data_time+nav_period+extra_nav_difs*node.packet.Tpream <= env.now:
						#in this case, there is no additional delay, we just go to start_nav
						print "node {} {}: received ValidHeader at {}, NAV period is included in listening period".format(node.nodeid, env.now, node.receive_data_time)
					else:							
						print "node {} {}: received ValidHeader at {} go into NAV({}) + [0,{}]{} DIFS until {}".format(node.nodeid, env.now, node.receive_data_time, nav_period, Wnav, extra_nav_difs, node.receive_data_time+nav_period+extra_nav_difs*node.packet.Tpream)
						#adjust to remove the extra time due to the fact that the data should have been received ealier					
						nav_period=nav_period+extra_nav_difs*node.packet.Tpream-(env.now-node.receive_data_time)					
						yield env.timeout(nav_period)		
				else:		
					#starts phase 3
					node.ca_state=start_phase3_backoff
					#random backoff [0,W3]
					node.backoff=random.randint(0,W3)
					print "node {} {}: start_phase3_backoff with W3={} backoff={} DIFS={}".format(node.nodeid, env.now, W3, node.backoff, node.packet.Tpream)
					#backoff period is backoff*DIFS, with DIFS=preamble duration
					yield env.timeout(node.backoff*node.packet.Tpream)

			###########################################################
			# start_phase2_listen -> start_nav (RTS)                  #
			###########################################################
			#node was in start_phase2_listen and it did receive an RTS
			#we process this event at the end of the listening period, normally the RTS has been received in the past
			if node.ca_state==start_phase2_listen and node.receive_rts==True:
				node.receive_rts = False
				node.total_listen_time = node.total_listen_time + (node.receive_rts_time - node.ca_listen_start_time)
				node.n_receive_nav_rts_p2 = node.n_receive_nav_rts_p2 + 1			
				#nav period is one listening period + W3*DIFS + TOA(data)
				nav_period=WL*node.packet.Tpream+node.packet.rectime + W3*node.packet.Tpream + airtime(node.packet.sf,node.packet.cr,node.nav,node.packet.bw)
				#add an additional number of random DIFS [0,Wnav]
				if Wnav!=0:
					extra_nav_difs=random.randint(0,Wnav)
				else:
					extra_nav_difs=0
				print "node {} {}: received RTS at {} go into NAV({}) + [0,{}]{} DIFS until {}".format(node.nodeid, env.now, node.receive_rts_time, nav_period, Wnav, extra_nav_difs, node.receive_rts_time+nav_period+extra_nav_difs*node.packet.Tpream)
				#adjust to remove the extra time due to the fact that the RTS should have been received ealier
				nav_period=nav_period+extra_nav_difs*node.packet.Tpream-(env.now-node.receive_rts_time)
				#go into NAV
				node.ca_state=start_nav			
				yield env.timeout(nav_period)

			###########################################################
			# start_phase3_backoff -> start_phase3_transmit           #
			###########################################################			
			if node.ca_state==start_phase3_backoff:		
				#we sent the DATA
				node.ca_state=start_phase3_transmit
				#change packet type to get the correct tine-on-air
				node.packet.setPacketType(dataPacketType)				

			###########################################################
			# start_phase3_transmit -> want_transmit | transmit DATA  #
			###########################################################				
			if node.ca_state==start_phase3_transmit and node.packet.ptype==dataPacketType:		
				channel_find_busy=False
			
				if check_busy:
					node.n_cca = node.n_cca + 1
					print "node {} {}: phase3 prior to send DATA checking channel".format(node.nodeid, env.now)
					#if channel is busy, then CCA reliability will decide if we can detect that channel is busy
					#if channel is not busy, as we observed no false positive, so channel_find_busy remains False
					if channel_busy_rts or channel_busy_data:
						print "node {}: channel is busy by {}".format(node.nodeid, 'RTS' if channel_busy_rts else 'DATA')
						if channel_busy_rts:
							node.n_busy_rts += 1
						else:
							node.n_busy_data += 1
						if random.randint(1,100) <= CCA_prob and CCA_prob!=0:
							channel_find_busy=True
							print "node {}: channel found busy by CCA with {}%".format(node.nodeid, CCA_prob)
						else:
							channel_find_busy=False
							print "node {}: channel found free by CCA".format(node.nodeid)		
					else:
						print "node {}: channel is free".format(node.nodeid)						
			
				#print "node {}: TEST -> force channel found free".format(node.nodeid)
				#channel_find_busy=False
			
				if channel_find_busy:
					node.cca=True
					print "node {}: number of retries left {}".format(node.nodeid, node.n_retry)
					node.n_retry = node.n_retry - 1							
					#and then we try again from the beginning of the CA procedure
					#we are not retrying several time the Wbusy procedure because if we reach this stage and channel is busy
					#it means that we lost priority, so better to start over again
					node.ca_state=want_transmit
				else:						 
					# DATA time sending and receiving
					# DATA packet arrives -> add to base station
					print "node {} {}: transmit DATA toa {} latency {} transmission ends at {}".format(node.nodeid, env.now, node.packet.rectime, env.now-node.want_transmit_time, env.now+node.packet.rectime)
					node.n_data_sent = node.n_data_sent + 1
					node.total_retry += n_retry - node.n_retry
					node.retry_bin[n_retry - node.n_retry] += 1
					node.latency = node.latency + (env.now-node.want_transmit_time)
					print "node {} : mean latency {}".format(node.nodeid, node.latency/node.n_data_sent)
					if (node in packetsAtBS):
						print "ERROR: DATA packet already in"
					else:
						if lora24GHz:
							sensitivity = sensi[node.packet.sf - 5, [203.125,406.25,812.5,1625].index(node.packet.bw) + 1]
						else:
							sensitivity = sensi[node.packet.sf - 6, [125,250,500].index(node.packet.bw) + 1]
						if node.packet.rssi < sensitivity:
							print "node {}: DATA packet will be lost".format(node.nodeid)
							node.packet.lost = True
						else:
							node.packet.lost = False
							checkcollision(node.packet)
							packetsAtBS.append(node)
							node.packet.addTime = env.now

					channel_busy_data=True
					yield env.timeout(node.packet.rectime)
					channel_busy_data=False
				
					if node.packet.lost:
						nrLost += 1
						print "node {} {}: DATA packet was lost".format(node.nodeid, env.now)
					if node.packet.collided == 1:
						nrCollisions = nrCollisions + 1
						print "node {} {}: DATA packet was collided".format(node.nodeid, env.now)
					if node.packet.collided == 0 and not node.packet.lost:
						nrReceived = nrReceived + 1
						print "node {} {}: DATA packet has been correctly transmitted".format(node.nodeid, env.now)
					if node.packet.processed == 1:
						nrProcessed = nrProcessed + 1

					# complete packet has been received by base station
					# can remove it
					if (node in packetsAtBS):
						packetsAtBS.remove(node)
					# reset the packet
					node.packet.collided = 0
					node.packet.processed = 0
					node.packet.lost = False
					node.n_retry=n_retry
					node.cca=False
					node.nav=0
					node.ca_state=schedule_tx

			###########################################################
			# start_nav -> want_transmit                              #
			###########################################################
			if node.ca_state==start_nav:
				#we arrive at the end of the nav period
				#so we try again from the beginning of the CA procedure
				node.ca_state=want_transmit
				node.packet.setPacketType(dataPacketType)
				print "node {} {}: number of retries left {}".format(node.nodeid, env.now, node.n_retry)
				node.n_retry = node.n_retry - 1	

		###////////////////////////////////////////////////////////				
		#no collision avoidance                                   /
		#original ALOHA-like behavior                             /
		###////////////////////////////////////////////////////////		
		else:

			if experiment==6:
				#normally 9 nodes with 100ms delay between each node
				transmit_wait=node.cycle*node.period-env.now+node.nodeid*100
			elif experiment==7:
				#normally 5 nodes with 500ms delay between each node
				transmit_wait=node.cycle*node.period-env.now+node.nodeid*500			
			else:
				transmit_wait = random.expovariate(1.0/float(node.period))

			print "node {} cycle {}: will try transmit in {} at {}".format(node.nodeid, node.cycle, transmit_wait, env.now+transmit_wait)

			node.cycle = node.cycle + 1
			
			yield env.timeout(transmit_wait)

			node.want_transmit_time=env.now
			
			n_transmit = n_transmit + 1
			if n_transmit > 1:
				current_inter_transmit_time = env.now - last_transmit_time
				inter_transmit_time += current_inter_transmit_time
				#put in bin from 0s to max_inter_transmit_time in second
				inter_transmit_time_bin[min(int(current_inter_transmit_time/1000), max_inter_transmit_time)] += 1	
			last_transmit_time = env.now				
			
			channel_find_busy=True
			
			while node.n_retry and channel_find_busy:
				if check_busy:
					node.n_cca = node.n_cca + 1
					print "node {} {}: noCA want_transmit checking channel".format(node.nodeid, env.now)
					#if channel is busy, then CCA reliability will decide if we can detect that channel is busy
					#if channel is not busy, as we observed no false positive, so channel_find_busy remains False
					if channel_busy_data:
						print "node {}: channel is busy".format(node.nodeid)
						node.n_busy_data += 1
						if random.randint(1,100) <= CCA_prob and CCA_prob!=0:
							channel_find_busy=True
							print "node {}: channel found busy by CCA with {}%".format(node.nodeid, CCA_prob)
						else:
							channel_find_busy=False
							print "node {}: channel found free by CCA".format(node.nodeid)		
					else:
						channel_find_busy=False
						print "node {}: channel is free".format(node.nodeid)						
				else:
					channel_find_busy=False
				
				#print "node {}: TEST -> force channel found free".format(node.nodeid)
				#channel_find_busy=False		
			
				if channel_find_busy:
					#here we just delay by a random backoff timer to retry again
					#random backoff [Wbusy_min,2**Wbusy_BE]
					node.backoff=random.randint(Wbusy_min,2**node.Wbusy_BE)
					print "node {}: channel found busy, backoff with Wbusy=[{},{}] backoff={} DIFS={}".format(node.nodeid, Wbusy_min, 2**node.Wbusy_BE, node.backoff, node.packet.Tpream)
					if Wbusy_exp_backoff:
						if node.Wbusy_BE<Wbusy_maxBE:
							node.Wbusy_BE=node.Wbusy_BE + 1
					print "node {}: number of retries left {}".format(node.nodeid, node.n_retry)
					node.n_retry = node.n_retry - 1
					if Wbusy_add_max_toa:			
						print "node {}: adding toa({})={}".format(node.nodeid, max_payload_size, airtime(node.packet.sf,node.packet.cr,max_payload_size,node.packet.bw))
						yield env.timeout(airtime(node.packet.sf,node.packet.cr,max_payload_size,node.packet.bw)+node.backoff*node.packet.Tpream)
					else:
						yield env.timeout(node.backoff*node.packet.Tpream)	

			if node.n_retry==0:
				print "node {} {}: current transmission aborted".format(node.nodeid, env.now)
				node.n_aborted = node.n_aborted +1
				node.n_retry=n_retry
				node.Wbusy_BE=Wbusy_BE
			else:	
				print "node {} {}: transmit DATA toa {} latency {} transmission ends at {}".format(node.nodeid, env.now, node.packet.rectime, env.now-node.want_transmit_time, env.now+node.packet.rectime)								
				node.n_data_sent = node.n_data_sent + 1
				node.total_retry += n_retry - node.n_retry
				node.retry_bin[n_retry - node.n_retry] += 1				
				node.latency = node.latency + (env.now-node.want_transmit_time)
				print "node {} : mean latency {}".format(node.nodeid, node.latency/node.n_data_sent)			
				if (node in packetsAtBS):
					print "ERROR: DATA packet already in"
				else:
					if lora24GHz:
						sensitivity = sensi[node.packet.sf - 5, [203.125,406.25,812.5,1625].index(node.packet.bw) + 1]
					else:
						sensitivity = sensi[node.packet.sf - 6, [125,250,500].index(node.packet.bw) + 1]
					if node.packet.rssi < sensitivity:
						print "node {}: DATA packet will be lost".format(node.nodeid)
						node.packet.lost = True
					else:
						node.packet.lost = False
						# adding packet if no collision
						if (checkcollision(node.packet)==1):
							node.packet.collided = 1
						else:
							node.packet.collided = 0
						packetsAtBS.append(node)
						node.packet.addTime = env.now

				channel_busy_data=True
				yield env.timeout(node.packet.rectime)
				channel_busy_data=False
		
				if node.packet.lost:
					nrLost += 1
				if node.packet.collided == 1:
					nrCollisions = nrCollisions + 1
				if node.packet.collided == 0 and not node.packet.lost:
					nrReceived = nrReceived + 1
					print "node {} {}: DATA packet has been correctly transmitted".format(node.nodeid, env.now)
				if node.packet.processed == 1:
					nrProcessed = nrProcessed + 1
			
				# complete packet has been received by base station
				# can remove it
				if (node in packetsAtBS):
					packetsAtBS.remove(node)
				# reset the packet
				node.packet.collided = 0
				node.packet.processed = 0
				node.packet.lost = False
				node.n_retry=n_retry
				node.Wbusy_BE=Wbusy_BE

		global targetSentPacket
		#sent = sum(n.n_data_sent for n in nodes)
		if nrProcessed > targetSentPacket:
			global endSim
			endSim=env.now
			return
		
		global lastDisplayTime	
		if nrProcessed % 10000 == 0 and env.now!=lastDisplayTime:
			print >> sys.stderr, nrProcessed, "-",
			lastDisplayTime=env.now	
#
# "main" program
#

# get arguments
if len(sys.argv) >= 6:
	CA = bool(int(sys.argv[1]))
	nrNodes = int(sys.argv[2])
	avgSendTime = int(sys.argv[3])
	experiment = int(sys.argv[4])
	simtime = int(sys.argv[5])
	if len(sys.argv) > 6:
		full_collision = bool(int(sys.argv[6]))	
	if CA:
		if len(sys.argv) > 7:
			WL = int(sys.argv[7])
		if len(sys.argv) > 8:
			W2 = int(sys.argv[8])				
			initialW2=W2			
		if len(sys.argv) > 9:
			W3 = int(sys.argv[9])								
		if len(sys.argv) > 10:
			Wnav = int(sys.argv[10])				
		if len(sys.argv) > 11:
			W2afterNAV = int(sys.argv[11])
		if len(sys.argv) > 12:
			P = int(sys.argv[12])				
			
	print "Nodes:", nrNodes
	print "AvgSendTime:", avgSendTime
	print "Distribution:", 'expoDistribType' if distribType==expoDistribType else 'uniformDistribType'
	print "Experiment:", experiment
	print "Simtime:", simtime
	print "Full Collision:", full_collision
	print "n_retry:", n_retry
	print "check_busy:", check_busy
	print "CCA_prob:", CCA_prob
	print "Packet length:", packetLength  
	print "max_payload_size:", max_payload_size
	print "targetSentPacket:", targetSentPacket
	print "Wbusy_min:", Wbusy_min
	print "Wbusy_BE:", Wbusy_BE
	print "Wbusy_maxBE:", Wbusy_maxBE
	print "Wbusy_exp_backoff:", Wbusy_exp_backoff
	print "Collision Avoidance:", CA

	if CA1:
		P=0
		#to keep the global amount of time for listening CA1_WL can be defined as 2*WL
		WL=CA1_WL
	if CA2:
		P=100	
		#to keep the global amount of time for listening CA2_WL can be defined as 2*WL		
		WL=CA2_WL
	if CA:
		print "P:", P
		print "WL:", WL		
		print "W2:", W2		
		print "W3:", W3			
		print "Wnav:", Wnav
		print "W2afterNAV:", W2afterNAV			
		print "n_retry_rts:", n_retry_rts
		print "check_busy_rts:", check_busy_rts					
			
else:
	print "usage: ./loraDir_mac <ca=0> <nodes> <avgsend> <experiment> <simtime> [collision]"
	print "usage: ./loraDir_mac <ca=1> <nodes> <avgsend> <experiment> <simtime> [collision] [WL] [W2] [W3] [Wnav] [W2afterNAV] [P]"	
	print "experiment 0 and 1 use 1 frequency only"
	exit(-1)

#raw_input('Press Enter to continue ...')

targetSentPacket = targetSentPacket * nrNodes
	
# global stuff
#Rnd = random.seed(12345)
nodes = []
packetsAtBS = []
env = simpy.Environment()

# maximum number of packets the BS can receive at the same time
maxBSReceives = 8

# max distance: 300m in city, 3000 m outside (5 km Utz experiment)
# also more unit-disc like according to Utz
bsId = 1
nrCollisions = 0
nrRTSCollisions = 0
nrReceived = 0
nrRTSReceived = 0
nrProcessed = 0
nrRTSProcessed = 0
nrLost = 0
nrRTSLost = 0

if lora24GHz:
	Ptx = 10
else:
	Ptx = 14
gamma = 2.08
d0 = 40.0
var = 0				# variance ignored for now
Lpld0 = 127.41
GL = 0

if lora24GHz:
	sensi = np.array([sf5,sf6,sf7,sf8,sf9,sf10,sf11,sf12])
	if experiment in [0,1,4,6,7]:
		minsensi = sensi[7,2]	 # 7th row is SF12, 2nd column is BW203
	elif experiment == 2:
		minsensi = sensi[0,5]	 # row 0 is SF5, 5th column is BW1625
	elif experiment in [3,5]:
		minsensi = np.amin(sensi) ## Experiment 3 can use any setting, so take minimum
else:
	sensi = np.array([sf6,sf7,sf8,sf9,sf10,sf11,sf12])
	if experiment in [0,1,4,6,7]:
		minsensi = sensi[6,2]	 # 6th row is SF12, 2nd column is BW125
	elif experiment == 2:
		minsensi = sensi[0,4]	 # first row is SF6, 4th column is BW500
	elif experiment in [3,5]:
		minsensi = np.amin(sensi) ## Experiment 3 can use any setting, so take minimum
	
Lpl = Ptx - minsensi
print "amin", minsensi, "Lpl", Lpl
maxDist = d0*(math.e**((Lpl-Lpld0)/(10.0*gamma)))
print "maxDist:", maxDist

# base station placement
bsx = maxDist+10
bsy = maxDist+10
xmax = bsx + maxDist + 20
ymax = bsy + maxDist + 20

# prepare graphics and add sink
if (graphics == 1):
	plt.ion()
	plt.figure()
	ax = plt.gcf().gca()
	# XXX should be base station position
	ax.add_artist(plt.Circle((bsx, bsy), 3, fill=True, color='green'))
	ax.add_artist(plt.Circle((bsx, bsy), maxDist, fill=False, color='green'))

if experiment==6:
	nrNodes=9

if experiment==7:
	nrNodes=5

#disable printing to stdout to make simulation much faster
if print_sim==False:
	f = open('/dev/null', 'w')
	sys.stdout = f	
			
for i in range(0,nrNodes):
	# myNode takes period (in ms), base station id packetlen (in Bytes)
	# 1000000 = 16 min
	node = myNode(i, endDeviceType, bsId, avgSendTime, distribType, packetLength)
	nodes.append(node)
	env.process(transmit(env,node))	
	print "-----------------------------------------------------------------------------------------------"
	
#prepare show
if (graphics == 1):
	plt.xlim([0, xmax])
	plt.ylim([0, ymax])
	plt.draw()
	plt.show()

# start simulation
env.run(until=simtime)

# compute energy
# Transmit consumption in mA from -2 to +17 dBm
# TODO for lora24GHz
TX = [22, 22, 22, 23,										 										# RFO/PA0: -2..1
			24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,	 	# PA_BOOST/PA1: 2..14
			82, 85, 90,											 											# PA_BOOST/PA1: 15..17
			105, 115, 125]										 										# PA_BOOST/PA1+PA2: 18..20

#use 5mA for receive consumption in mA. This can be achieve by SX126X LoRa chip
RX = 5
V = 3.3		# voltage XXX

#restore printing
sys.stdout=stdout_print_target

#statistic per node
#
for node in nodes:
	print "-- node {} ------------------------------------------------------------------".format(node.nodeid)
	print "number of CAD:", node.n_cca
	#normally it is 2 and 4 symbols, but in reality it is closer to 3 and 5 symbols
	nCadSym=3
	if node.packet.sf > 8:
		nCadSym=nCadSym+2
	#for lora24GHz we use 4 symbols for CAD	
	if lora24GHz:
		nCadSym=4	
	#consumption must be converted into mA: cad_consumption[node.packet.sf-7]/1e6	
	energy = (node.packet.symTime * (cad_consumption[node.packet.sf-7]/1e6) * V * node.n_cca * nCadSym) / 1e6
	node.cca_energy=energy
	print "energy in CAD (in J):", energy									
	energy = (node.data_rectime * TX[int(node.packet.txpow)+2] * V * node.n_data_sent \
							+ node.rts_rectime * TX[int(node.packet.txpow)+2] * V * node.n_rts_sent) / 1e6
	print "energy in transmission (in J):", energy
	if CA:
		energy = (node.total_listen_time * RX * V) / 1e6
		print "energy in listening (in J):", energy
	print "total energy (in J):", (node.data_rectime * TX[int(node.packet.txpow)+2] * V * node.n_data_sent \
							+ node.rts_rectime * TX[int(node.packet.txpow)+2] * V * node.n_rts_sent \
							+ node.total_listen_time * RX * V) / 1e6 + node.cca_energy
	print "end of simulation time {}ms {}h".format(endSim, float(endSim/3600000))
	print "cumulated time (s) in TX:", (node.data_rectime*node.n_data_sent+node.rts_rectime*node.n_rts_sent)/1000
	if CA:
		print "cumulated time (s) in RX:", node.total_listen_time/1000		
	print "sent data packets:", node.n_data_sent	
	print "mean latency:", node.latency/node.n_data_sent
	print "aborted packets:", node.n_aborted
	print "mean retry:", node.total_retry/node.n_data_sent
	print "retry distribution:"
	print node.retry_bin
	print "retry sum:", sum(node.retry_bin)
	for i in range(0,n_retry):
		s = sum(node.retry_bin[0:i+1])
		print "%.1f" % (s*100.0/sum(node.retry_bin)),
	print ""		
	print "channel busy DATA:", node.n_busy_data
	if CA:
		print "channel busy RTS:", node.n_busy_rts
		print "channel busy RTS (P1):", node.n_busy_rts_p1		
		print "sent rts packets:", node.n_rts_sent	
		print "NAV from RTS P1:", node.n_receive_nav_rts_p1
		print "NAV from RTS P2:", node.n_receive_nav_rts_p2
		print "NAV from RTS ++:", node.n_receive_nav_rts_p1+node.n_receive_nav_rts_p2		
		print "NAV from DATA P1:", node.n_receive_nav_data_p1	
		print "NAV from DATA P2:", node.n_receive_nav_data_p2
		print "NAV from DATA ++:", node.n_receive_nav_data_p1+node.n_receive_nav_data_p2	
		print "RTS retry distribution:"
		print node.retry_rts_bin
		print "rts retry sum:", sum(node.retry_rts_bin)
		for i in range(0,n_retry_rts):
			s = sum(node.retry_rts_bin[0:i+1])
			print "%.1f" % (s*100.0/sum(node.retry_rts_bin)),
		print ""
			
for i in range(0,2):
	if i==1:
		fname = "exp" + str(experiment) + ".dat"
		f=open(fname, 'a')
		# Change the standard output to the file we created
		sys.stdout = f 
	
	print "-- SETTINGS -----------------------------------------------------------------"

	print "Nodes:", nrNodes
	print "AvgSendTime:", avgSendTime
	print "Distribution:", 'expoDistribType' if distribType==expoDistribType else 'uniformDistribType'
	print "Experiment:", experiment
	print "Simtime:", simtime
	print "Full Collision:", full_collision
	print "Toa DATA:", nodes[0].data_rectime
	print "Toa RTS:", nodes[0].rts_rectime	
	print "DIFS:", nodes[0].packet.Tpream	
	print "n_retry:", n_retry
	print "check_busy:", check_busy
	print "CCA_prob:", CCA_prob
	print "Packet length:", packetLength  
	print "targetSentPacket:", targetSentPacket
	print "Wbusy_min:", Wbusy_min
	print "Wbusy_BE:", Wbusy_BE
	print "Wbusy_maxBE:", Wbusy_maxBE
	print "Wbusy_exp_backoff:", Wbusy_exp_backoff
	print "Collision Avoidance:", CA
	if CA:
		print "P:", P
		print "WL:", WL		
		print "W2:", W2		
		print "W3:", W3			
		print "Wnav:", Wnav
		print "W2afterNAV:", W2afterNAV			
		print "n_retry_rts:", n_retry_rts
		print "check_busy_rts:", check_busy_rts			
    	
	print "-- TOTAL --------------------------------------------------------------------"
	
	sent = sum(n.n_data_sent for n in nodes)
	rts_sent = sum(n.n_rts_sent for n in nodes)
	n_receive_nav_data_p1 = sum(n.n_receive_nav_data_p1 for n in nodes)
	n_receive_nav_data_p2 = sum(n.n_receive_nav_data_p2 for n in nodes)	
	n_receive_nav_rts_p1 = sum(n.n_receive_nav_rts_p1 for n in nodes)
	n_receive_nav_rts_p2 = sum(n.n_receive_nav_rts_p2 for n in nodes)	
	energy = sum( n.data_rectime * TX[int(n.packet.txpow)+2] * V * n.n_data_sent \
								+ n.rts_rectime * TX[int(n.packet.txpow)+2] * V * n.n_rts_sent for n in nodes) / 1e6
	print "energy in CAD (in J):", sum( n.cca_energy	for n in nodes)						
	print "energy in transmission (in J):", energy
	energy = sum( n.total_listen_time * RX * V for n in nodes) / 1e6
	print "energy in listening (in J):", energy
	print "total energy (in J):", sum( n.data_rectime * TX[int(n.packet.txpow)+2] * V * n.n_data_sent \
								+ n.rts_rectime * TX[int(n.packet.txpow)+2] * V * n.n_rts_sent \
								+ n.total_listen_time * RX * V for n in nodes) / 1e6 + sum( n.cca_energy for n in nodes)
	print "end of simulation time {}ms {}h".format(endSim, float(endSim/3600000))							
	print "cumulated time (s) in TX:", sum( (n.data_rectime*n.n_data_sent+n.rts_rectime*n.n_rts_sent) for n in nodes)/1000
	if CA:
		print "cumulated time (s) in RX:", sum( (n.total_listen_time) for n in nodes)/1000
	print "number of CCA:", sum (n.n_cca for n in nodes)			
	print "sent data packets:", sent
	print "mean latency:", sum (float(n.latency)/float(n.n_data_sent) for n in nodes) / nrNodes
	print "aborted packets:", sum (n.n_aborted for n in nodes)
	print "collisions:", nrCollisions
	print "received packets:", nrReceived
	print "processed packets:", nrProcessed
	print "lost packets:", nrLost

	retry_bin = []

	print "retry distribution:"
		
	for node in nodes:
		print node.retry_bin

	for i in range(0,n_retry):
		s = 0
		for node in nodes:
			s = s + node.retry_bin[i]
		retry_bin.append(s)

	print "mean retry:", sum((float(n.total_retry)/float(n.n_data_sent)) for n in nodes)/nrNodes

	print retry_bin
	
	for i in range(0,n_retry):			
		s = sum(retry_bin[0:i+1])
		print "%.1f" % (s*100.0/sent),
		if s==sent:
			break
	print ""	
	
	print "channel busy DATA:", sum (n.n_busy_data for n in nodes)
	if CA:
		print "channel busy RTS:", sum (n.n_busy_rts for n in nodes)
		print "channel busy RTS (P1):", sum (n.n_busy_rts_p1 for n in nodes)		
		print "sent rts packets:", rts_sent	
		print "RTS collisions:", nrRTSCollisions
		print "RTS received packets:", nrRTSReceived
		print "RTS processed packets:", nrRTSProcessed
		print "RTS lost packets:", nrRTSLost
		print "NAV from RTS P1:", n_receive_nav_rts_p1
		print "NAV from RTS P2:", n_receive_nav_rts_p2
		print "NAV from RTS ++:", n_receive_nav_rts_p1+n_receive_nav_rts_p2		
		print "NAV from DATA P1:", n_receive_nav_data_p1	
		print "NAV from DATA P2:", n_receive_nav_data_p2
		print "NAV from DATA ++:", n_receive_nav_data_p1+n_receive_nav_data_p2	

		retry_rts_bin = []

		print "RTS retry distribution:"
		
		for node in nodes:
			print node.retry_rts_bin

		for i in range(0,n_retry_rts+1):
			s = 0
			for node in nodes:
				s = s + node.retry_rts_bin[i]
			retry_rts_bin.append(s)

		print "mean RTS retry:", sum((float(n.total_retry_rts)/float(n.n_rts_sent)) for n in nodes)/nrNodes

		print retry_rts_bin
	
		for i in range(0,n_retry_rts+1):			
			s = sum(retry_rts_bin[0:i+1])
			print "%.1f" % (s*100.0/rts_sent),
			if s==rts_sent:
				break
		print ""			

	if sent>0:
		# data extraction rate
		der = (sent-nrCollisions)/float(sent)
		print "DER:", der
		der = (nrReceived)/float(sent)
		print "DER method 2:", der
	
	print "n_transmit:", n_transmit	
	print "mean inter-transmit time (ms):", inter_transmit_time/float(n_transmit)
	print "inter-transmit time distribution [<1s, <2s, <3s, <4s, ...]:"
	print inter_transmit_time_bin
	
print "-- END ----------------------------------------------------------------------"	

"""	
# this can be done to keep graphics visible
if (graphics == 1):
	raw_input('Press Enter to continue ...')

# save experiment data into a dat file that can be read by e.g. gnuplot
# name of file would be:	exp0.dat for experiment 0
fname = "exp" + str(experiment) + ".dat"
print fname
if os.path.isfile(fname):
	res = "\n" + str(nrNodes) + " " + str(nrCollisions) + " "	 + str(sent) + " " + str(energy)
else:
	res = "#nrNodes nrCollisions nrTransmissions OverallEnergy\n" + str(nrNodes) + " " + str(nrCollisions) + " "	+ str(sent) + " " + str(energy)
with open(fname, "a") as myfile:
	myfile.write(res)
myfile.close()

"""
# with open('nodes.txt','w') as nfile:
#		for n in nodes:
#			nfile.write("{} {} {}\n".format(n.x, n.y, n.nodeid))
# with open('basestation.txt', 'w') as bfile:
#		bfile.write("{} {} {}\n".format(bsx, bsy, 0))
