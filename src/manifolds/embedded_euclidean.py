## This file is part of Jax Geometry
#
# Copyright (C) 2021, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/jaxgeometry
#
# Jax Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jax Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Theano Geometry. If not, see <http://www.gnu.org/licenses/>.
#


from src.setup import *
from src.params import *

from src.manifolds.manifold import *

from src.plotting import *
import matplotlib.pyplot as plt

class Euclidean_2d_3d(EmbeddedManifold):
    """ Euclidean space """

    def __init__(self):
        self.dim = 2
        self.emb_dim = 2

        self.update_coords = lambda coords,_: coords

        ##### Metric:
        self.g = lambda x: jnp.eye(self.emb_dim)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        
        F = lambda x: jnp.array([x[0][0], x[0][1], 0.0])
        invF = lambda x: jnp.array([x[0][0], x[0][1]])
    
        EmbeddedManifold.__init__(self,F,2,3,invF=invF)

    def __str__(self):
        return "Embedded Euclidean manifold of dimension %d" % (self.emb_dim)

    
