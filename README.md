# Gravitational N-Body Simulation

## Overview
This project is a gravitational n-body simulation using HTML5 Canvas and JavaScript. It employs the Barnes-Hut algorithm and a quadtree data structure to efficiently calculate gravitational forces and simulate the dynamic evolution of celestial bodies.

## Features
- **Gravitational Physics:** Customizable gravitational parameters, including gravitational constant scaling and softening.
- **Simulation Control:** Time step control, quadtree depth, and approximation thresholds are adjustable.
- **Celestial Evolution:** Features include planet and sun accretion thresholds, supernova mechanics, and black hole formation.
- **Rendering:** Options to toggle quadtree, center of mass, particle trails, and customize colors and sizes of celestial bodies.

## Key Parameters
- `G_SCALER`, `G`: Control the gravitational constant for simulation.
- `ETA`: Gravitational softening parameter.
- `DT`: Simulation timestep.
- `MAX_DEPTH`: Maximum quadtree depth.
- `THETA`: Barnes-Hut approximation threshold.

## Getting Started
1. Clone the repository.
2. Open `index.html` in a modern web browser to run the simulation.
3. Or, visit the Pages deploymnt [here](https://derbydefi.github.io/n-body/)

## Controls
- **Zoom/Pan:** Use mouse wheel to zoom; drag to pan.
- **UI Controls:** Adjust simulation parameters in real-time via the UI.

## Development
- **Add Particles:** Select a particle type from the dropdown and click on the simulation area to add particles.
- **Simulation Settings:** Tweak parameters like gravitational constant, timestep, and quadtree depth for different behaviors.

## Technical Details
- **Quadtree:** Used to partition the space to reduce the complexity of force calculations. In this implementation we add a dynamic boundary growth.
- **Barnes-Hut Algorithm:** Approximates distant particles as a single mass to speed up force computation.
- **Euler Method:** integration method used for simplicity, can be optimized futher with Runge-Kutta or Symplectic integrator techniques
- 
## License
public domain :)
with love from derby

