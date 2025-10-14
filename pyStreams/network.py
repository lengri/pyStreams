import numpy as np
import matplotlib.pyplot as plt

class Channel:
    
    def __init__(
        self,
        x: np.ndarray,
        K_sp: float|int,
        m_sp: float|int,
        n_sp: float|int,
        uplift_rate: float|int,
        drainage_area: np.ndarray|tuple = (2, 1.7),
        min_drainage_area: float|int = 1e6,
        z: np.ndarray|None = None,
        run_initial_spinup: bool = True
    ):
        
        """
        Initialises a Channel instance.
        
        Parameters:
        -----------
            x: np.ndarray
                Along-stream distances to the outlet.
            K_sp: float|int
                Erodibility constant of the SPL.
            m_sp: float|int
                Area exponent of the SPL.
            n_sp: float|int
                Slope exponent of the SPL.
            uplift_rate: float|int
                Uplift rate.
            drainage_area: np.ndarray|tuple
                Drainage area for each node of the channel. Can either be an array of
                shape x.shape, or a tuple representing Hack's Law parameters. 
            min_drainage_area: float|int
                Drainage area that flows into the channel head.
            z: np.ndarray|None
                Initial elevation values, np.zero_like(x, dtype=float) by default.
            run_initial_spinup: bool
                Option to run the model upon initialisation until a steady state is achieved.
                
        Returns:
        --------
            None
        """
        if z is None:
            self.z = np.zeros_like(x, dtype=float)
        else:
            self.z = z 
        
        # If A is a tuple, use a form of Hack's Law to establish drainage areas
        if type(drainage_area) is tuple:
            drainage_area = drainage_area[0]*(x.max()-x)**drainage_area[1] + min_drainage_area
            
        self.x = x.copy()
        self.dx = x[1]-x[0]
        self.A = drainage_area
        self.K_sp = K_sp 
        self.m_sp = m_sp
        self.n_sp = n_sp
        self.U = uplift_rate
        self.minA = min_drainage_area
        
        if run_initial_spinup:
            while True:
                z_old = self.z[-1]
                self.run_one_step(
                    uplift_rate=self.U,
                    drainage_area=self.A,
                    K_sp=self.K_sp,
                    n_sp=self.n_sp,
                    m_sp=self.m_sp,
                    dt=1000
                )
                if np.abs(1-z_old/self.z[-1]) < 0.00001: # arbitrary stopping condition for steady state, TODO: adjust this!
                    break
        
    def change_drainage_area_by_fraction(
        self,
        frac_per_dt: float
    ):
        """
        Reduce the drainage area by a set fraction (for every dt!).
        
        Parameters:
        -----------
            frac_per_dt: float
                Fractional change to be imposed at the next time step.
                For example, 0.01 will reduce all drainage areas by 1% between
                the current and next time step.
        
        Returns:
        --------
            None
        """
        # You can do this by hand and supply the change to Channel.run_one_step(drainage_area=...)
        # Or just call this function to change A?
        
        # Since we don't know the time step, you have to figure than one out yourself
        self.A -= self.A*frac_per_dt
    
    def run_one_step(
        self,
        dt: float|int,
        uplift_rate: float|int|None = None,
        drainage_area: np.ndarray|tuple|None = None,
        K_sp: float|int|None = None,
        n_sp: float|int|None = None,
        m_sp: float|int|None = None
    ):
        """
        Evolve the channel by one step. If parameters are defined as None, the instance will use
        the parameters of the last time step when parameters were defined. These may be inherited from 
        the instance initialisation.
        
        Parameters:
        -----------
            dt: float|int
                Time step size.
            uplift_rate: float|int
                Uplift rate.
            drainage_area: np.ndarray|tuple|None
                Option to adjust the drainage area from its previous values. Can either be an array of
                shape x.shape, or a tuple representing Hack's Law parameters. A simple fractional
                change in drainage area can be achieved by using Channel.change_drainage_area_by_fraction().
            K_sp: float|int|None
                Erodibility constant of the Stream Power Law.
            n_sp: float|int|None
                Slope exponent of the SPL.
            m_sp: float|int|None
                Drainage area exponent of the SPL.
        
        Returns:
        --------
            None
        """
        # Check if the inputs are not None...
        if uplift_rate is not None: self.U = uplift_rate
        if drainage_area is not None:
            if type(drainage_area) is tuple:
                self.A = drainage_area[0]*(self.x.max()-self.x)**drainage_area[1] + self.minA
            else:
                self.A = drainage_area
        if K_sp: self.K_sp = K_sp 
        if n_sp: self.n_sp = n_sp 
        if m_sp: self.m_sp = m_sp
        
        # we keep the base node at constant elevation!
        erosion = self.K_sp * (self.A[1:] ** self.m_sp) * ((self.z[1:] - self.z[:-1]) / self.dx) * dt
        self.z[1:] = self.z[1:] + (self.U * dt - erosion)

class Network:
    
    def __init__(
        self,
        trunk: Channel,
    ):
        """
        Initialises a Network instance.
        
        Parameters:
        -----------
            trunk: Channel
                The main trunk represented by a Channel instance.
                
        Returns:
        --------
            None
        """
        self.trunk = trunk
        self.tributaries = {}
    
    def _fix_tributary_elevations(self):
        # fix all the tribuary elevations to be at the right heights
        for node, channel in self.tributaries.items():
            channel.z -= (channel.z[0] - self.trunk.z[node])
        
    def attach_tributary(
        self,
        node_id: int,
        channel_instance: Channel
    ):
        """
        Places a triburary at the defined node along the main channel. 
        Automatically matches elevations and upstream distances.
        
        Parameters:
        -----------
            node_id: int
                Id of node along the main Channel (Network.trunk)
            channel_instance: Channel
                A Channel instance representing the tributary.
        """
        
        # alter the channel instance such that x is in relative coordinates along the trunk
        channel_instance.x += self.trunk.x[node_id]
        
        # add instance to the internal dict
        self.tributaries[node_id] = channel_instance
        
        self._fix_tributary_elevations()
        
    def evolve_network(
        self,
        dt: float|int,
        uplift_rate: float|int|None = None,
        K_sp: float|int|None = None,
        n_sp: float|int|None = None,
        m_sp: float|int|None = None
    ):
        """
        Evolve the trunk stream and tributaries of the network for one time step.
        If all input parameters (except dt) are set to None, evolve_network will
        use the parameters internal to the respective Channel instances.
        Evolution of tributaries is controlled by incision at the base node connecting to
        the trunk stream.
        
        Parameters:
        -----------
            dt: float|int
                Time step size.
            uplift_rate: float|int
                Uplift rate.
            K_sp: float|int|None
                Erodibility constant of the Stream Power Law.
            n_sp: float|int|None
                Slope exponent of the SPL.
            m_sp: float|int|None
                Drainage area exponent of the SPL.
        
        Returns:
        --------
            None
        """
        
        # if any of the supplied parameters are none, apply the ones internal to
        # the Channel instances
        
        # evolve the main stem and record how much incision occurs
        z_old = np.array([self.trunk.z[id] for id in self.tributaries.keys()])
        self.trunk.run_one_step(
            dt=dt,
            uplift_rate=uplift_rate,
            K_sp=K_sp,
            n_sp=n_sp,
            m_sp=m_sp
        )
        z_new = np.array([self.trunk.z[id] for id in self.tributaries.keys()])
        
        incision = dt*self.trunk.U-(z_new-z_old)
        
        # step through the tributaries and evolve each by incising the base
        for i, (_, trib) in enumerate(self.tributaries.items()):
            trib.z[0] -= incision[i] # incise the base of the tributary
            # evolve one step
            trib.run_one_step(
                dt=dt,
                uplift_rate=0, # no uplift, just the base incision
                K_sp=K_sp,
                n_sp=n_sp,
                m_sp=m_sp
            )
            # adjust the outlet to be on the same elevation as the trunk node
        self._fix_tributary_elevations()
        
    
    def show_network(self, ax=None):
        pass
    
if __name__ == "__main__":
    
    ## There is some slighly weird behaviour here where using the same xtrib for 
    ## both Channel instances and altering those x values will also alter them in other 
    ## Channel instances. Solution: copy the x array that is input?
    
    dx = 10
    xtrunk = np.arange(0, 3000, dx)
    xtrib1 = np.arange(0, 1500, dx)
    xtrib2 = np.arange(0, 1000, dx)
    K_sp = 1e-6
    m_sp = 0.5
    n_sp = 1
    U = 1e-3
    dt = 500
    
    trunk = Channel(
        x=xtrunk,
        K_sp=K_sp,
        m_sp=m_sp,
        n_sp=n_sp,
        uplift_rate=U,
        drainage_area=(2, 2)
    )
    
    trib1 = Channel(
        x=xtrib1,
        K_sp=K_sp,
        m_sp=m_sp,
        n_sp=n_sp,
        uplift_rate=U,
        drainage_area=(2, 1.8)
    )

    trib2 = Channel(
        x=xtrib2,
        K_sp=K_sp,
        m_sp=m_sp,
        n_sp=n_sp,
        uplift_rate=U,
        drainage_area=(2, 1.5)
    )
    
    network = Network(
        trunk=trunk
    )
    
    network.attach_tributary(
        node_id=int(len(xtrunk)/3),
        channel_instance=trib1
    )
    
    network.attach_tributary(
        node_id=2*int(len(xtrunk)/3),
        channel_instance=trib2
    )
    
    fg, ax = plt.subplots(1, 2)
    
    # Drainage areas in initial state
    ax[0].plot(network.trunk.x, network.trunk.A)
    for node, channel in network.tributaries.items():
        ax[0].plot(channel.x, channel.A)
        
    # Channel profiles in initial state
    ax[1].plot(network.trunk.x, network.trunk.z)
    for node, channel in network.tributaries.items():
        ax[1].plot(channel.x, channel.z)
        
    # Run the model with increased uplift rates to create a knickpoint.
    for i in range(0, 1100):
        network.evolve_network(
            uplift_rate=2e-3,
            dt=1000
        )

    ax[1].plot(network.trunk.x, network.trunk.z)

    for node, channel in network.tributaries.items():
        ax[1].plot(channel.x, channel.z)
    plt.show()
    