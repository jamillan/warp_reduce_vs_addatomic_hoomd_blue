from __future__ import print_function, division
import sys
sys.path.append('/projects/b1030/hoomd/hoomd-2.1.9-new_walls/')
sys.path.append('/projects/b1030/hoomd/hoomd-2.1.9-new_walls/hoomd')
#sys.path.append('/home/jaime/software/hoomd-install')
#sys.path.append('/home/jaime/software/hoomd-install/hoomd')

import os
import random
#os.environ['LD_LIBRARY_PATH']  = "/home/jaimemillan/boost/lib" 
#print os.environ['LD_LIBRARY_PATH']



from hoomd import *
from hoomd.md import *
import hoomd.deprecated as deprecated
import numpy

c=context.initialize()

if len(option.get_user()) == 0:
 workspace = '.';
else:
 workspace = option.get_user()[0]



system = deprecated.init.read_xml("250000.xml")
system.replicate(nx=2,ny=1,nz=1)
group_A = group.type(name='a-particles', type = 'A')

#introduce SLJ forces between particles to make them 'hard'
nl =nlist.cell()
c.sorter.disable()
slj= pair.slj(r_cut = 3.5, nlist=nl)
slj.set_params(mode="shift")
slj.pair_coeff.set('A','A' , epsilon=1.0, sigma=1.0 )
slj.pair_coeff.set('A','B' , epsilon=0.0, sigma=1.0 )
slj.pair_coeff.set('B','B' , epsilon=0.0, sigma=1.0 )

#introduce Yukawa Walls bounding from  above/below Z direction

sigma=1.0
walls = wall.group()
walls.set_tag(0);
wall_pos = 0.5*system.box.Lz - 1
walls.add_plane((0,0,wall_pos),(0.,0.,-1.))
walls.add_plane((0,0,-wall_pos),(0.,0.,1.))
wall_force_slj=wall.lj(walls, r_cut=4.0,active_planes=[0,1])
wall_force_slj.force_coeff.set('A', epsilon= 1.5,r_cut=4,sigma=sigma,r_extrap = 0.05)
wall_force_slj.force_coeff.set('B', epsilon= 0,r_cut=4,sigma=sigma,r_extrap = 0.05)


for p in system.particles:
 vx = (2.0 * random.random()- 1.0 )
 vy = (2.0 * random.random()- 1.0 )
 vz = (2.0 * random.random()- 1.0 )
 p.velocity = (vx,vy,vz)
 p.diameter = 1.0


#log Thermos
logger = analyze.log(quantities=['temperature' , 'potential_energy', 'kinetic_energy'],
												 period=5e2, filename='log.log', overwrite=True)



#Create Trajectory

integrate.mode_standard(dt=0.001)

#NVE Integration
integrator = integrate.nve(group_A , limit = 0.0001)
zero = update.zero_momentum(period =100)

run(1e3)
integrator.disable()
zero.disable()



#NVT interation to reached target temperature

tf=0.01
integrator = integrate.nvt(group=group_A , tau = 0.65 , kT = 0.001)
integrator.set_params(kT=variant.linear_interp(points=[(0, logger.query('temperature')), (2e6, 0.75)]))

run(2e6)

integrator.set_params(kT=variant.linear_interp(points=[(0, logger.query('temperature')), (2e6, tf)]))
run(2e6)


# start benchmark
tps = benchmark.series(warmup=0, repeat=4, steps=70000, limit_hours=20.0/3600.0)
ptps = numpy.average(tps) * len(system.particles);

print("Hours to complete 10e6 steps: {0}".format(10e6/(ptps/len(system.particles))/3600));
meta.dump_metadata(filename = workspace+"/metadata.json", user = {'mps': ptps, 'tps': tps});


