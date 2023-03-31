#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:10:14 2023

@author: fmry
"""

#%% Sources

#%% Modules

#JAX
import jax
from jax import vmap
import jax.numpy as jnp
import optax
import jaxopt
import haiku as hk
import tensorflow as tf
import tensorflow_datasets as tfds

#Numpy
import numpy as np

#JAXGeometry
from src.stochastics import product_sde
from src.stochastics.product_sde import tile

#%% Standard Functions

def get_coords(Fx, M):
    chart = M.centered_chart(Fx)
    return (M.invF((Fx,chart)),chart)

def to_TM(Fx,v, M):
    x = get_coords(Fx)
#     return jnp.dot(M.JF(x),jnp.dot(M.invJF((Fx,x[1])),v))
    JFx = M.JF(x)
    return jnp.dot(JFx,jnp.linalg.lstsq(JFx,v)[0])

#%% Fixed x0 and t

def fixed_x0_t(M, x0, T, dts, dWs, net_f = None, opt = None, epochs = 20, batch_size = 2**5, 
               sample_size = 2**9, 
               max_iter = None,
               print_step = 10,
               save_step = 10):
    
    def net_default(x):
        """Create model."""
        model = hk.Sequential([
            hk.Linear(20), jax.nn.tanh,
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(M.emb_dim),
        ])

        y = model(x)
        return y
    
    def pre_update(params,state, *ars,**kwargs):
        
        if save_bool:
            if state.iter_num % save_step == 0:
                loss.append(state.value)
                params_lst.append(params)
            
        if print_bool:
            if state.iter_num % print_step == 0:
                print(f"[Step {state.iter_num}], epoch {state.iter_num//(sample_size//batch_size)}, training loss: {state.value:.3f}.")
                
        return params,state

    #Fixed x0 and t
    def generator():
        while True:
            """Generates batches of samples."""
            N = batch_size
            (ts,xss,chartss,*_) = product(tile(x0,N),
                                          _dts,
                                          dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))
            samples = xss[-1]
            charts = chartss[-1] 

            yield vmap(lambda x,chart: M.F((x,chart)))(samples,charts)

    def loss_fun(params, data):
        """ compute loss."""
        s = net.apply(params, data)
        (xs,charts) = vmap(lambda x: get_coords(x,M))(data)
        norm2s = vmap(lambda s: jnp.sum(jnp.square(s)))(s)
        divs = vmap(lambda x,chart,s: M.div((x,chart),
                                                lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                    net.apply(params,M.F(x))))
                       )(xs,charts,s)

        return jnp.mean(norm2s+2.0*divs)
    
    if max_iter is None:
        max_iter = epochs * sample_size // batch_size
    if net_f is None:
        net_f = net_default
    if opt is None:
        opt = optax.sgd(0.01)
        
    if print_step is None:
        print_bool = False
    else:
        print_bool = True
        
    if save_step is None:
        save_bool = False
    else:
        save_bool = True
        
    (product,sde_product,chart_update_product) = product_sde.initialize(M,M.sde_Brownian_coords,M.chart_update_Brownian_coords)
    _dts = dts(T=T)
    
    solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=max_iter, pre_update=pre_update)
    
    ds = tf.data.Dataset.from_generator(generator,output_types=tf.float32,output_shapes=([batch_size,M.emb_dim]))
    ds = iter(tfds.as_numpy(ds))
    
    net = hk.without_apply_rng(hk.transform(net_f))
    params = net.init(jax.random.PRNGKey(42), next(ds))
    
    loss = []
    params_lst = []
    
    state = solver.init_state(params)
    params, state = solver.run_iterator(
        init_params=params, iterator=ds)
    
    if save_bool:
        return loss, params_lst, params, state, net
    else:
        return params, state, net
    
#%% Fixed x0

def fixed_x0(M, x0, T, dts, dWs, net_f = None, opt = None, epochs = 20, 
             repeats = 2**5, ts_per_batch = 2**4,
               sample_size = 2**9, 
               max_iter = None,
               print_step = 10,
               save_step = 10):
    
    def net_default(x):
        """Create model."""
        model = hk.Sequential([
            hk.Linear(20), jax.nn.tanh,
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(M.emb_dim),
        ])

        y = model(x)
        return y
    
    def pre_update(params,state, *ars,**kwargs):
        
        if save_bool:
            if state.iter_num % save_step == 0:
                loss.append(state.value)
                params_lst.append(params)
            
        if print_bool:
            if state.iter_num % print_step == 0:
                print(f"[Step {state.iter_num}], epoch {state.iter_num//(sample_size//batch_size)}, training loss: {state.value:.3f}.")
                
        return params,state

    #Fixed x0 and t
    def generator():
        while True:
            N = repeats
            (ts,xss,chartss,*_) = product(x0s,_dts,dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))        
            inds = np.random.choice(range(_dts.shape[0]),ts_per_batch,replace=False)
    #         inds = jnp.arange(0,ts.shape[0]+1,ts.shape[0]//ts_per_batch)[1:]-1
    #         inds = jnp.array([49,99])
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            yield jnp.hstack((jax.vmap(lambda x,chart: M.F((x,chart)))(samples.reshape((-1,M.dim)),charts.reshape((-1,chartss.shape[-1]))),
                             jnp.repeat(ts.reshape((-1,1)),N,axis=1).reshape((-1,1))
                            ))

    def loss_fun(params, data):
        
        s = net.apply(params,data)
        Fxts = data[:,0:M.emb_dim]
        ts = data[:,-1]
        (xts,chartts) = jax.vmap(lambda x: get_coords(x, M))(Fxts)
        norm2s = jax.vmap(lambda s: jnp.sum(jnp.square(s)))(s)
        divs = jax.vmap(lambda xt,chartt,t: M.div((xt,chartt),
                                                  lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                    net.apply(params,jnp.hstack((M.F(x),t)))))
                       )(xts,chartts,ts)
        return jnp.mean(norm2s+2.0*divs)
    
    batch_size = ts_per_batch*repeats
    if max_iter is None:
        max_iter = epochs * sample_size // batch_size
    if net_f is None:
        net_f = net_default
    if opt is None:
        opt = optax.sgd(0.01)
        
    if print_step is None:
        print_bool = False
    else:
        print_bool = True
        
    if save_step is None:
        save_bool = False
    else:
        save_bool = True
        
    (product,sde_product,chart_update_product) = product_sde.initialize(M,M.sde_Brownian_coords,M.chart_update_Brownian_coords)
    _dts = dts(T=T)
    x0s = tile(x0,repeats)
    
    solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=max_iter, pre_update=pre_update)
    
    ds = tf.data.Dataset.from_generator(generator,output_types=tf.float32,output_shapes=([batch_size,M.emb_dim+1]))
    ds = iter(tfds.as_numpy(ds))
    
    net = hk.without_apply_rng(hk.transform(net_f))
    params = net.init(jax.random.PRNGKey(42), next(ds))
    
    loss = []
    params_lst = []
    
    state = solver.init_state(params)
    params, state = solver.run_iterator(
        init_params=params, iterator=ds)
    
    if save_bool:
        return loss, params_lst, params, state, net
    else:
        return params, state, net
    
#%% Fixed t

def fixed_t(M, x0, T, dts, dWs, net_f = None, opt = None, epochs = 20, 
            samples_per_x0 = 2**5, batch_size = 2**8,
               sample_size = 2**12, 
               max_iter = None,
               print_step = 10,
               save_step = 10):
    
    def net_default(x):
        """Create model."""
        model = hk.Sequential([
            hk.Linear(20), jax.nn.tanh,
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(M.emb_dim),
        ])

        y = model(x)
        return y
    
    def pre_update(params,state, *ars,**kwargs):
        
        if save_bool:
            if state.iter_num % save_step == 0:
                loss.append(state.value)
                params_lst.append(params)
            
        if print_bool:
            if state.iter_num % print_step == 0:
                print(f"[Step {state.iter_num}], epoch {state.iter_num//(sample_size//batch_size)}, training loss: {state.value:.3f}.")
                
        return params,state

    #Fixed x0 and t
    def generator():
        while True:
            """Generates batches of samples."""
            N = batch_size
            global x0s
            (ts,xss,chartss,*_) = product(x0s,_dts,dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))
            samples = xss[-1]
            charts = chartss[-1]
            Fx0s = jax.vmap(lambda x,chart: M.F((x,chart)))(*x0s)
            x0s = (jnp.repeat(samples[::samples_per_x0],samples_per_x0,axis=0),
                   jnp.repeat(charts[::samples_per_x0],samples_per_x0,axis=0))
            yield jnp.hstack((Fx0s,
                              vmap(lambda x,chart: M.F((x,chart)))(samples,charts)))
            
    def loss_fun(params, data):
        """ compute loss."""
        s = net.apply(params,data)
        Fx0s = data[:,0:M.emb_dim]
        Fxts = data[:,M.emb_dim:]
        (xts,chartts) = jax.vmap(lambda x: get_coords(x,M))(Fxts)
        norm2s = jax.vmap(lambda s: jnp.sum(jnp.square(s)))(s)
        divs = jax.vmap(lambda Fx0,xt,chartt: M.div((xt,chartt),
                                                    lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                      net.apply(params,jnp.hstack((Fx0,M.F(x))))))
                       )(Fx0s,xts,chartts)
        return jnp.mean(norm2s+2.0*divs)
    
    if max_iter is None:
        max_iter = epochs * sample_size // batch_size
    if net_f is None:
        net_f = net_default
    if opt is None:
        opt = optax.sgd(0.01)
        
    if print_step is None:
        print_bool = False
    else:
        print_bool = True
        
    if save_step is None:
        save_bool = False
    else:
        save_bool = True
        
    global x0s
    x0s = tile(x0,batch_size)
    (product,sde_product,chart_update_product) = product_sde.initialize(M,M.sde_Brownian_coords,M.chart_update_Brownian_coords)
    _dts = dts(T=T)
    
    solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=max_iter, pre_update=pre_update)
    
    ds = tf.data.Dataset.from_generator(generator,output_types=tf.float32,output_shapes=([batch_size,2*M.emb_dim]))
    ds = iter(tfds.as_numpy(ds))
    
    net = hk.without_apply_rng(hk.transform(net_f))
    params = net.init(jax.random.PRNGKey(42), next(ds))
    
    loss = []
    params_lst = []
    
    state = solver.init_state(params)
    params, state = solver.run_iterator(
        init_params=params, iterator=ds)
    
    if save_bool:
        return loss, params_lst, params, state, net
    else:
        return params, state, net
    
#%% General

def score_matching(M, x0, T, dts, dWs, net_f = None, opt = None, epochs = 20, 
            samples_per_x0 = 2**5, repeats = 2**3, ts_per_batch = 2**5,
               sample_size = 2**13, 
               max_iter = None,
               print_step = 10,
               save_step = 10):
    
    def net_default(x):
        """Create model."""
        model = hk.Sequential([
            hk.Linear(20), jax.nn.tanh,
            hk.Linear(10), jax.nn.tanh,
            hk.Linear(M.emb_dim),
        ])

        y = model(x)
        return y
    
    def pre_update(params,state, *ars,**kwargs):
        
        if save_bool:
            if state.iter_num % save_step == 0:
                loss.append(state.value)
                params_lst.append(params)
            
        if print_bool:
            if state.iter_num % print_step == 0:
                print(f"[Step {state.iter_num}], epoch {state.iter_num//(sample_size//batch_size)}, training loss: {state.value:.3f}.")
                
        return params,state

    #Fixed x0 and t
    def generator():
        while True:
            """Generates batches of samples."""
            N = samples_per_x0*repeats
            _dts = dts(T=1.)
            global x0s
            (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],samples_per_x0,axis=0),jnp.repeat(x0s[1],samples_per_x0,axis=0)),
                                          _dts,dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))
            Fx0s = jax.vmap(lambda x,chart: M.F((x,chart)))(*x0s)
            x0s = (xss[-1,::samples_per_x0],chartss[-1,::samples_per_x0])
            inds = np.random.choice(range(_dts.shape[0]),ts_per_batch,replace=False)
    #         inds = jnp.arange(0,ts.shape[0]+1,ts.shape[0]//ts_per_batch)[1:]-1
    #         inds = jnp.array([-1,-1])
            ts = ts[inds]
            samples = xss[inds]
            charts = chartss[inds]
            yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,samples_per_x0,axis=0),(ts_per_batch,1)),
                             jax.vmap(lambda x,chart: M.F((x,chart)))(samples.reshape((-1,M.dim)),charts.reshape((-1,chartss.shape[-1]))),
                             jnp.repeat(ts,N).reshape((-1,1))
                            ))
            
    def loss_fun(params, data):
        """ compute loss."""
        s = net.apply(params,data)
        Fx0s = data[:,0:M.emb_dim]
        Fxts = data[:,M.emb_dim:2*M.emb_dim]
        ts = data[:,-1]
        (xts,chartts) = jax.vmap(lambda x: get_coords(x,M))(Fxts)
        norm2s = jax.vmap(lambda s: jnp.sum(jnp.square(s)))(s)
        divs = jax.vmap(lambda Fx0,xt,chartt,t: M.div((xt,chartt),
                                                      lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                        net.apply(params,jnp.hstack((Fx0,M.F(x),t)))))
                       )(Fx0s,xts,chartts,ts)
        return jnp.mean(norm2s+2.0*divs)
    
    batch_size = samples_per_x0*ts_per_batch*repeats
    if max_iter is None:
        max_iter = epochs * sample_size // batch_size
    if net_f is None:
        net_f = net_default
    if opt is None:
        opt = optax.sgd(0.01)
        
    if print_step is None:
        print_bool = False
    else:
        print_bool = True
        
    if save_step is None:
        save_bool = False
    else:
        save_bool = True
        
    global x0s
    x0s = tile(x0,repeats)
    (product,sde_product,chart_update_product) = product_sde.initialize(M,M.sde_Brownian_coords,M.chart_update_Brownian_coords)
    _dts = dts(T=T)
    
    solver = jaxopt.OptaxSolver(opt=opt, fun=loss_fun, maxiter=max_iter, pre_update=pre_update)
    
    ds = tf.data.Dataset.from_generator(generator,output_types=tf.float32,output_shapes=([batch_size,2*M.emb_dim+1]))
    ds = iter(tfds.as_numpy(ds))
    
    net = hk.without_apply_rng(hk.transform(net_f))
    params = net.init(jax.random.PRNGKey(42), next(ds))
    
    loss = []
    params_lst = []
    
    state = solver.init_state(params)
    params, state = solver.run_iterator(
        init_params=params, iterator=ds)
    
    if save_bool:
        return loss, params_lst, params, state, net
    else:
        return params, state, net

"""
#Fixed x0
def generator():
    while True:
        N = repeats
        _dts = dts(T=1.)
        global x0s
        (ts,xss,chartss,*_) = product(x0s,_dts,dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))        
        inds = np.random.choice(range(_dts.shape[0]),ts_per_batch,replace=False)
#         inds = jnp.arange(0,ts.shape[0]+1,ts.shape[0]//ts_per_batch)[1:]-1
#         inds = jnp.array([49,99])
        ts = ts[inds]
        samples = xss[inds]
        charts = chartss[inds]
        yield jnp.hstack((jax.vmap(lambda x,chart: M.F((x,chart)))(samples.reshape((-1,M.dim)),charts.reshape((-1,chartss.shape[-1]))),
                         jnp.repeat(ts.reshape((-1,1)),N,axis=1).reshape((-1,1))
                        ))
        
def loss_fun(params, data):
    s = net.apply(params,data)
    Fxts = data[:,0:M.emb_dim]
    ts = data[:,-1]
    (xts,chartts) = jax.vmap(get_coords)(Fxts)
    norm2s = jax.vmap(lambda s: jnp.sum(jnp.square(s)))(s)
    divs = jax.vmap(lambda xt,chartt,t: M.div((xt,chartt),
                                              lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                net.apply(params,jnp.hstack((M.F(x),t)))))
                   )(xts,chartts,ts)
    return jnp.mean(norm2s+2.0*divs)

#Fixed t
def generator():
    while True:
        N = batch_size
        _dts = dts(T=.5)
        global x0s
        (ts,xss,chartss,*_) = product(x0s,_dts,dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))
        samples = xss[-1]
        charts = chartss[-1]
        Fx0s = jax.vmap(lambda x,chart: M.F((x,chart)))(*x0s)
        x0s = (jnp.repeat(samples[::samples_per_x0],samples_per_x0,axis=0),
               jnp.repeat(charts[::samples_per_x0],samples_per_x0,axis=0))
        yield jnp.hstack((Fx0s,
                          jax.vmap(lambda x,chart: M.F((x,chart)))(samples,charts)))
        
def loss_fun(params, data):
    s = net.apply(params,data)
    Fx0s = data[:,0:M.emb_dim]
    Fxts = data[:,M.emb_dim:]
    (xts,chartts) = jax.vmap(get_coords)(Fxts)
    norm2s = jax.vmap(lambda s: jnp.sum(jnp.square(s)))(s)
    divs = jax.vmap(lambda Fx0,xt,chartt: M.div((xt,chartt),
                                                lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                  net.apply(params,jnp.hstack((Fx0,M.F(x))))))
                   )(Fx0s,xts,chartts)
    return jnp.mean(norm2s+2.0*divs)

#General
def generator():
    while True:
        N = samples_per_x0*repeats
        _dts = dts(T=1.)
        global x0s
        (ts,xss,chartss,*_) = product((jnp.repeat(x0s[0],samples_per_x0,axis=0),jnp.repeat(x0s[1],samples_per_x0,axis=0)),
                                      _dts,dWs(N*M.dim,_dts).reshape(-1,N,M.dim),jnp.repeat(1.,N))
        Fx0s = jax.vmap(lambda x,chart: M.F((x,chart)))(*x0s)
        x0s = (xss[-1,::samples_per_x0],chartss[-1,::samples_per_x0])
        inds = np.random.choice(range(_dts.shape[0]),ts_per_batch,replace=False)
#         inds = jnp.arange(0,ts.shape[0]+1,ts.shape[0]//ts_per_batch)[1:]-1
#         inds = jnp.array([-1,-1])
        ts = ts[inds]
        samples = xss[inds]
        charts = chartss[inds]
        yield jnp.hstack((jnp.tile(jnp.repeat(Fx0s,samples_per_x0,axis=0),(ts_per_batch,1)),
                         jax.vmap(lambda x,chart: M.F((x,chart)))(samples.reshape((-1,M.dim)),charts.reshape((-1,chartss.shape[-1]))),
                         jnp.repeat(ts,N).reshape((-1,1))
                        ))
        
def loss_fun(params, data):
    s = net.apply(params,data)
    Fx0s = data[:,0:M.emb_dim]
    Fxts = data[:,M.emb_dim:2*M.emb_dim]
    ts = data[:,-1]
    (xts,chartts) = jax.vmap(get_coords)(Fxts)
    norm2s = jax.vmap(lambda s: jnp.sum(jnp.square(s)))(s)
    divs = jax.vmap(lambda Fx0,xt,chartt,t: M.div((xt,chartt),
                                                  lambda x: jnp.dot(M.invJF((M.F(x),x[1])),
                                                                    net.apply(params,jnp.hstack((Fx0,M.F(x),t)))))
                   )(Fx0s,xts,chartts,ts)
    return jnp.mean(norm2s+2.0*divs)

"""