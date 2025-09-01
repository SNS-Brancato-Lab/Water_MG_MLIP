from ase import units
from ase.md.nptberendsen import NPTBerendsen
from ase.md import velocitydistribution
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
import numpy as np
import os


#from ipi.interfaces.sockets import SocketClient
from ase.calculators.socketio import SocketClient

from mace.calculators import MACECalculator

# Load the MACE model
mace_calculator = MACECalculator(model_paths='water_QE_revPBE_TOTAL_D3_r65_128-01_2_2_stagetwo.model', device='cuda')

# Load the initial configuration with PBC
init_conf = read('water_start.xyz', '0')
init_conf.set_pbc(True)
init_conf.set_calculator(mace_calculator)

# Retrieve the i-PI server hostname and port from environment variables
host = os.environ.get('IPI_HOST')
port = int(os.environ.get('IPI_PORT', '31415'))

if host is None:
    raise ValueError("Environment variable 'IPI_HOST' is not set.")

print(f"Connecting to i-PI server at {host}:{port}")

# Create Client using TCP/IP sockets
client = SocketClient(host=host, port=port)

# Run the client
client.run(init_conf, use_stress=True)



