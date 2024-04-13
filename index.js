//gravity physics
let G_SCALER = 1e5;
let G = 6.6743e-11 * G_SCALER; // Gravitational constant adjusted for sim

let ETA = 30; // gravitational softening--deals with "too close encounters"

//time and simulation control
let DT = 1; // timestep
let MAX_DEPTH = 14; //max amount of divisions in quadtree
let THETA = 0.4; // Barnes-Hut approximation threshold

//accretion and celestial evolution
let PLANET_ACCRETION_THRESHOLD = 5; // how many particles it takes to make a planet
let SUN_ACCRETION_THRESHOLD = 10; // how many particles it takes to make a sun
let SUN_TO_SUPERNOVA_AGE = 1000; // represents simulation steps or units of time before supernova could happen
let ACCRETION_CHANCE = 0.1; // Chance of accretion happening upon close encounter
let SUPERNOVA_CHANCE = 0.1; // Chance of a sun going supernova after reaching the supernova age
let MAX_EXPLOSION_SPEED = 10; // velocity of emitted particles duiring supernova explosion effect
let BLACKHOLE_CHANCE = 0.01; // chance a supernova leaves a blackhole
let BLACKHOLE_ACCRETION_CHANCE = 0.1; //chance a black hole will remove another particle
let BLACKHOLE_EVENT_HORIZON = 1; // radius multiplier
let MAX_MASS = 1e10; // An arbitrary large mass for black holes
let SUPERNOVA_PARTICLE_GEN_MULTIPLIER = 2; // creates new particles in supernovae

//initial conditons
let INITIAL_PARTICLE_COUNT = 5000;
let INITIAL_PARTICLE_MASS = 1e6; // minimum mass of particles
let INITAL_VELOCITY = 20; // vel for initalizations
let RADIUS_ROTATION = 200; // radius for initialiazitions

//rendering/visuals
let RENDER_QUADTREE = false; //doesnt play nice with panning/zooming
let RENDER_CENTER_OF_MASS = false;
let PARTICLE_COLOR_SLOW = [173, 216, 230];
let PARTICLE_COLOR_FAST = [0, 0, 139];
let PLANET_COLOR = "lime";
let SUN_COLOR = "yellow";
let BLACKHOLE_COLOR = "maroon";
let PARTICLE_RADIUS = 1; //rendering radius
let PLANET_RADIUS = 1; //rendering radius
let SUN_RADIUS = 2; //rendering radius
let BLACKHOLE_RADIUS = 3; //rendering radius
let MAX_TRAIL = 2; // max length of tracer trails
let RENDER_TRAIL = true; // renders tracers for particles
let TRAIL_ALPHA = 0.2; // 0 to 1 to adjust trails transparency

class Particle {
	constructor(x, y, mass, type = "particle") {
		this.x = x;
		this.y = y;
		this.mass = mass;
		this.vx = 0;
		this.vy = 0;
		this.fx = 0;
		this.fy = 0;
		this.type = type;
		this.age = 0;
		this.accretedParticles = []; // Track accreted particles for potential supernova explosion
		this.accretionCount = 0;
		this.isMerged = false;
		this.trail = [];

		this.updateRadius();
	}

	applyForce(fx, fy) {
		this.fx += fx;
		this.fy += fy;
	}
	incrementAge() {
		this.age += 1;
	}
	update(dt) {
		//why isnt this being used
		// Update velocity based on the force applied.
		this.vx += (this.fx / this.mass) * dt;
		this.vy += (this.fy / this.mass) * dt;

		// Update position based on velocity.
		this.x += this.vx * dt;
		this.y += this.vy * dt;

		// Reset the force.
		this.fx = 0;
		this.fy = 0;
		if (RENDER_TRAIL) {
			if (this.trail.length >= MAX_TRAIL) {
				this.trail.shift(); // Remove the oldest position
			}
			this.trail.push({ x: this.x, y: this.y });
		}
	}
	mergeWith(otherParticle) {
		// Calculate the new mass.
		const newMass = this.mass + otherParticle.mass;

		// Calculate the new velocity using conservation of momentum: p = m * v.
		const newVx =
			(this.vx * this.mass + otherParticle.vx * otherParticle.mass) / newMass;
		const newVy =
			(this.vy * this.mass + otherParticle.vy * otherParticle.mass) / newMass;

		// Update position to be the mass-weighted average of the two particles' positions.
		this.x =
			(this.x * this.mass + otherParticle.x * otherParticle.mass) / newMass;
		this.y =
			(this.y * this.mass + otherParticle.y * otherParticle.mass) / newMass;

		// Update this particle with the new properties.
		this.mass = newMass;
		this.vx = newVx;
		this.vy = newVy;
		this.accretionCount += 1;
		this.updateRadius();

		// Reset the forces since they'll need to be recalculated next update.
		this.fx = 0;
		this.fy = 0;

		otherParticle.isMerged = true; // Mark otherParticle as merged/absorbed.

		// Note: OtherParticle is considered merged/absorbed, and should be removed from the simulation's particle list.
		this.accretedParticles.push(otherParticle);
	}
	accreteIfClose(otherParticle) {
		if (this.type === "blackhole") {
			if (Math.random() < BLACKHOLE_ACCRETION_CHANCE) {
				return false;
			}
			const dx = otherParticle.x - this.x;
			const dy = otherParticle.y - this.y;
			const distance = Math.sqrt(dx * dx + dy * dy + ETA);

			// Define the accretion radius for the black hole
			const accretionRadius = this.radius * BLACKHOLE_EVENT_HORIZON; // Example: 5 times the radius of the black hole

			if (distance < accretionRadius) {
				this.mass += otherParticle.mass; // Increase the black hole's mass
				// Momentum conservation to find new velocity if necessary
				this.vx =
					(this.vx * this.mass + otherParticle.vx * otherParticle.mass) /
					(this.mass + otherParticle.mass);
				this.vy =
					(this.vy * this.mass + otherParticle.vy * otherParticle.mass) /
					(this.mass + otherParticle.mass);
				return true; // Return true if the particle is accreted
			}
		}
		return false;
	}

	updateRadius() {
		// Adjust radius for visibility using a simplified scaling, ensuring visibility for all mass ranges
		this.radius = Math.log(this.mass) / Math.log(10) - 4; // Example scaling formula for visualization
		//this.radius = Math.max(this.radius, 1); // Ensure a minimum radius for visibility
		//console.log("new radius =", this.radius);
	}
}

class QuadtreeNode {
	constructor(boundary, depth = 0) {
		this.boundary = boundary; // { x, y, width, height }
		this.depth = depth;
		this.particles = [];
		this.children = [];
		this.centerOfMass = { x: 0, y: 0 };
		this.totalMass = 0;
		this.isDivided = false;
		//this.maxNodes = 1000
	}

	insert(particle) {
		if (!this.isInside(particle)) {
			if (!this.expandBoundaries(particle)) {
				return false; // Expansion failed or not allowed, particle not inserted
			}
		}

		if (!this.isDivided && this.particles.length < 1) {
			this.particles.push(particle);
			return true;
		}

		if (!this.isDivided) {
			this.subdivide();
		}

		let inserted = this.children.some((child) => child.insert(particle));
		if (!inserted) {
			this.particles.push(particle); // Keep in this node if it doesn't fit any child
		}

		return true;
	}

	static countNodes(node) {
		if (!node.isDivided) {
			return 1;
		}
		return (
			1 +
			node.children.reduce(
				(acc, child) => acc + QuadtreeNode.countNodes(child),
				0
			)
		);
	}
	getRootNode() {
		let currentNode = this;
		while (currentNode.parent) {
			currentNode = currentNode.parent;
		}
		return currentNode;
	}
	isInside(particle) {
		return (
			particle.x >= this.boundary.x &&
			particle.x <= this.boundary.x + this.boundary.width &&
			particle.y >= this.boundary.y &&
			particle.y <= this.boundary.y + this.boundary.height
		);
	}
	expandBoundaries(particle) {
		if (this.depth !== 0) {
			// Only the root node should expand
			return false;
		}

		// Calculate the necessary expansion to include the particle
		let growthFactor = 1.1; // Expand boundaries by 10%
		let minX = Math.min(this.boundary.x, particle.x);
		let maxX = Math.max(this.boundary.x + this.boundary.width, particle.x);
		let minY = Math.min(this.boundary.y, particle.y);
		let maxY = Math.max(this.boundary.y + this.boundary.height, particle.y);

		// Calculate new boundaries with minimal necessary growth to include the particle
		let newWidth = (maxX - minX) * growthFactor;
		let newHeight = (maxY - minY) * growthFactor;
		let newX = minX - (newWidth - (maxX - minX)) / 2;
		let newY = minY - (newHeight - (maxY - minY)) / 2;

		// Set the new boundaries
		this.boundary = { x: newX, y: newY, width: newWidth, height: newHeight };

		// Reinsert all particles into the new tree only if expansion has occurred
		let particles = this.collectParticles();
		this.clear();
		particles.forEach((p) => this.insert(p));

		return true;
	}

	collectParticles() {
		let particles = [];
		if (this.isDivided) {
			this.children.forEach((child) => {
				particles = particles.concat(child.collectParticles());
			});
		} else {
			return this.particles.slice(); // Return a copy of the array
		}
		return particles;
	}

	clear() {
		this.particles = [];
		this.children = [];
		this.isDivided = false;
	}

	calculateMassDistribution() {
		if (!this.isDivided) {
			if (this.particles.length === 1) {
				this.centerOfMass = { x: this.particles[0].x, y: this.particles[0].y };
				this.totalMass = this.particles[0].mass;
			}
			return;
		}

		let totalX = 0,
			totalY = 0;
		this.totalMass = 0;

		this.children.forEach((child) => {
			child.calculateMassDistribution();
			this.totalMass += child.totalMass;
			totalX += child.centerOfMass.x * child.totalMass;
			totalY += child.centerOfMass.y * child.totalMass;
		});

		if (this.totalMass > 0) {
			this.centerOfMass.x = totalX / this.totalMass;
			this.centerOfMass.y = totalY / this.totalMass;
		}
	}

	subdivide() {
		if (this.depth >= MAX_DEPTH) {
			// Prevent further subdivision beyond the maximum depth
			return;
		}
		const rootNode = this.getRootNode(); // Method to retrieve the root node
		const currentNodeCount = QuadtreeNode.countNodes(rootNode);
		//console.log(currentNodeCount);

		const { x, y, width, height } = this.boundary;
		const halfWidth = width / 2;
		const halfHeight = height / 2;

		this.children[0] = new QuadtreeNode(
			{ x: x + halfWidth, y: y, width: halfWidth, height: halfHeight },
			this.depth + 1
		);
		this.children[1] = new QuadtreeNode(
			{ x: x, y: y, width: halfWidth, height: halfHeight },
			this.depth + 1
		);
		this.children[2] = new QuadtreeNode(
			{ x: x, y: y + halfHeight, width: halfWidth, height: halfHeight },
			this.depth + 1
		);
		this.children[3] = new QuadtreeNode(
			{
				x: x + halfWidth,
				y: y + halfHeight,
				width: halfWidth,
				height: halfHeight,
			},
			this.depth + 1
		);
		this.isDivided = true;

		//console.log("Subdivided node at", this.boundary);

		this.children.forEach((child) =>
			this.particles.forEach((particle) => child.insert(particle))
		);

		this.particles = [];
	}
}

class Simulation {
	constructor(width = 800, height = 600) {
		this.width = width;
		this.height = height;
		this.particles = [];
		this.currentMaxVelocity = 0;
		this.dynamicDt = true;
		this.currentMaxAcceleration = 0;
		this.dt = DT; // Time step
		this.planetCount = 0;
		this.sunCount = 0;
		this.blackHoleCount = 0;
		this.supernovaCount = 0;
		// Define the boundary based on provided dimensions
		this.boundary = { x: 0, y: 0, width: this.width, height: this.height };
	}
	updateMaxVelocity() {
		let maxVelocity = 0;
		this.particles.forEach((particle) => {
			const velocityMagnitude = Math.sqrt(particle.vx ** 2 + particle.vy ** 2);
			if (velocityMagnitude > maxVelocity) {
				maxVelocity = velocityMagnitude;
			}
		});
		this.currentMaxVelocity = maxVelocity.toFixed(4);
	}
	updateMaxAcceleration() {
		let maxAcceleration = 0;
		this.particles.forEach((particle) => {
			// Calculate acceleration magnitude from force components and particle mass
			// Acceleration a = F/m where F is force and m is mass
			const accMagnitude = Math.sqrt(
				(particle.fx ** 2 + particle.fy ** 2) / (particle.mass * particle.mass)
			);
			if (accMagnitude > maxAcceleration) {
				maxAcceleration = accMagnitude;
			}
		});
		this.currentMaxAcceleration = maxAcceleration.toFixed(4); // Store the maximum acceleration with precision
	}

	update() {
		this.rootNode = new QuadtreeNode(this.boundary);
		// Initialize the quadtree for this update cycle.

		// Insert particles into the quadtree.
		this.particles.forEach((particle) => this.rootNode.insert(particle));

		// Calculate forces using the quadtree to optimize the process.
		this.calculateForcesWithQuadtree();
		this.handleAccretionAndBlackHoleInteractions();

		// Handle accretion and potential particle mergers.
		this.handleAccretion();

		// Update particle positions.
		this.particles.forEach((particle) => particle.update(this.dt));
		this.updateMaxVelocity();

		// Remove merged particles.
		//this.particles = this.particles.filter((particle) => !particle.isMerged);

		// Update celestial bodies (if any).
		this.updateCelestialBodies();
	}

	calculateForcesWithQuadtree() {
		this.rootNode.calculateMassDistribution(); // Ensure mass distribution is up-to-date.

		this.particles.forEach((particle) => {
			this.calculateForceForParticle(particle, this.rootNode);
		});
	}

	calculateForceForParticle(particle, node) {
		if (!node.isDivided || node.particles.length === 1) {
			// Direct force calculation for nodes without subdivisions
			node.particles.forEach((otherParticle) => {
				// Ensure the particle is different and exists
				if (particle !== otherParticle && otherParticle !== undefined) {
					let dx = otherParticle.x - particle.x;
					let dy = otherParticle.y - particle.y;
					let distance = Math.sqrt(dx * dx + dy * dy + ETA * ETA);
					if (distance > 0) {
						// Avoid division by zero
						let force =
							(G * particle.mass * otherParticle.mass) / (distance * distance);
						let fx = (force * dx) / distance;
						let fy = (force * dy) / distance;
						particle.applyForce(fx, fy);
					}
				}
			});
		} else {
			// Simplified calculation for distant particles, using the center of mass
			let dx = node.centerOfMass.x - particle.x;
			let dy = node.centerOfMass.y - particle.y;
			let distance = Math.sqrt(dx * dx + dy * dy + ETA * ETA);

			// Check if node's center of mass and total mass are defined
			if (node.centerOfMass.x !== undefined && node.totalMass !== undefined) {
				// Determine whether to use the simplified force calculation
				let s = Math.max(node.boundary.width, node.boundary.height);
				if (s / distance < THETA) {
					if (distance > 0) {
						// Avoid division by zero
						let force =
							(G * particle.mass * node.totalMass) / (distance * distance);
						let fx = (force * dx) / distance;
						let fy = (force * dy) / distance;
						particle.applyForce(fx, fy);
					}
				} else {
					// Recursively calculate force for children nodes
					node.children.forEach((child) =>
						this.calculateForceForParticle(particle, child)
					);
				}
			}
		}
	}

	handleAccretion() {
		let particlesToRemove = new Set();

		const checkAndMergeParticles = (node) => {
			//if (node.particles.length < 2) return; // No possibility of accretion in this node.
			//console.log(node.particles.length);
			for (let i = 0; i < node.particles.length; i++) {
				for (let j = i + 1; j < node.particles.length; j++) {
					let p1 = node.particles[i];
					let p2 = node.particles[j];
					let dx = p1.x - p2.x;
					let dy = p1.y - p2.y;
					let distance = Math.sqrt(dx * dx + dy * dy + ETA);

					if (
						distance < p1.radius + p2.radius + ETA * ETA &&
						Math.random() < ACCRETION_CHANCE
					) {
						// Assuming radius is a property or calculated from mass.
						// Accretion occurs only if the random check passes.
						//console.log(`here`);
						p1.mergeWith(p2);
						particlesToRemove.add(p2); // Mark particle for removal.
					}
				}
			}

			// Recursively check children if this node is divided

			if (node.isDivided) {
				node.children.forEach(checkAndMergeParticles);
			}
		};

		// Start the accretion check from the root node.
		checkAndMergeParticles(this.rootNode);

		// Filter out merged particles.
		this.particles = this.particles.filter((p) => !particlesToRemove.has(p));
	}
	handleAccretionAndBlackHoleInteractions() {
		let particlesToRemove = new Set();

		for (let i = 0; i < this.particles.length; i++) {
			let p1 = this.particles[i];
			if (p1.type === "blackhole") {
				for (let j = 0; j < this.particles.length; j++) {
					if (i !== j) {
						let p2 = this.particles[j];
						if (p1.accreteIfClose(p2)) {
							particlesToRemove.add(p2);
						}
					}
				}
			}
		}

		// Remove accreted particles
		this.particles = this.particles.filter((p) => !particlesToRemove.has(p));
	}

	updateCelestialBodies() {
		let particlesToAdd = [];
		let particlesToRemove = new Set();

		this.particles.forEach((particle) => {
			if (particle.type !== "particle") {
				particle.incrementAge();
			}

			// Handle transitions based on accretion count
			if (
				particle.type === "particle" &&
				particle.accretionCount >= PLANET_ACCRETION_THRESHOLD
			) {
				if (particle.type === "particle") {
					this.planetCount++;
				}
				particle.type = "planet";
				particle.mass = INITIAL_PARTICLE_MASS * particle.accretionCount;
			} else if (
				(particle.type === "planet" || particle.type === "particle") &&
				particle.accretionCount >= SUN_ACCRETION_THRESHOLD
			) {
				if (particle.type === "planet") {
					this.planetCount--;
				}
				this.sunCount++;
				particle.type = "sun";
				particle.mass = INITIAL_PARTICLE_MASS * particle.accretionCount;
				particle.age = 0;
			} else if (
				particle.type === "sun" &&
				particle.age > SUN_TO_SUPERNOVA_AGE
			) {
				if (Math.random() < SUPERNOVA_CHANCE) {
					this.supernovaCount++;
					this.sunCount--;
					const emittedParticleCount =
						particle.accretionCount * SUPERNOVA_PARTICLE_GEN_MULTIPLIER;
					for (let i = 0; i < emittedParticleCount; i++) {
						const angle = Math.random() * 2 * Math.PI;
						const speed = Math.random() * MAX_EXPLOSION_SPEED;
						const vx = speed * Math.cos(angle);
						const vy = speed * Math.sin(angle);
						const emittedParticle = new Particle(
							particle.x,
							particle.y,
							INITIAL_PARTICLE_MASS,
							"particle"
						);
						emittedParticle.vx = vx;
						emittedParticle.vy = vy;
						particlesToAdd.push(emittedParticle);
					}
					particle.accretedParticles = [];
					if (Math.random() < BLACKHOLE_CHANCE) {
						particle.type = "blackhole";
						particle.mass = MAX_MASS;
						this.blackHoleCount++;
					} else {
						particlesToRemove.add(particle);
					}
				}
			}
		});

		this.particles.push(...particlesToAdd);
		this.particles = this.particles.filter((p) => !particlesToRemove.has(p));
	}

	randomizeParticles(count) {
		for (let i = 0; i < count; i++) {
			const x = Math.random() * this.width;
			const y = Math.random() * this.height;
			const mass = INITIAL_PARTICLE_MASS;
			const particle = new Particle(x, y, mass, "particle", this);
			particle.vx = (Math.random() - 0.5) * 1; // Adjust magnitude as needed
			particle.vy = (Math.random() - 0.5) * 1;
			this.particles.push(particle);
		}
	}

	randomizeRotatingDisk(
		count,
		diskRadius,
		velocityMagnitude = INITAL_VELOCITY
	) {
		const centerX = this.width / 2;
		const centerY = this.height / 2;
		const mass = INITIAL_PARTICLE_MASS; // Constant mass for each particle, can be randomized as needed

		for (let i = 0; i < count; i++) {
			// Uniformly distribute angles for disk rotation
			const angle = Math.random() * 2 * Math.PI;
			// Randomly distribute particles within the disk radius
			const radius = Math.sqrt(Math.random()) * diskRadius; // sqrt for uniform distribution

			// Convert polar coordinates (radius, angle) to Cartesian coordinates (x, y)
			const x = centerX + radius * Math.cos(angle);
			const y = centerY + radius * Math.sin(angle);

			// Create particle with position
			const particle = new Particle(x, y, mass, "particle", this);

			// Assign velocities for circular motion
			// Adjusting velocity magnitude can control the rotation speed and direction
			// Here, we give particles a velocity perpendicular to the line connecting them to the center, simulating circular motion
			// Velocity magnitude inversely proportional to sqrt(radius) simulates Keplerian rotation
			const speed = velocityMagnitude / Math.sqrt(radius); // Adjust base speed as needed
			particle.vx = -speed * Math.sin(angle); // Perpendicular to radius, "-" for clockwise
			particle.vy = speed * Math.cos(angle); // Perpendicular to radius

			this.particles.push(particle);
		}
	}
	randomizeBinaryStarSystem(count, starMass = 1e8, separation = 300) {
		const centerX = this.width / 2;
		const centerY = this.height / 2;

		// Place two large central masses
		const star1 = new Particle(
			centerX - separation / 2,
			centerY,
			starMass,
			"sun",
			this
		);
		const star2 = new Particle(
			centerX + separation / 2,
			centerY,
			starMass,
			"sun",
			this
		);
		star1.vx = 0.5;
		star1.vy = 0;
		star2.vx = -0.5;
		star2.vy = 0;
		this.sunCount = 2;
		this.particles.push(star1, star2);

		// Add smaller particles orbiting around the binary stars
		for (let i = 0; i < count; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const radius = Math.sqrt(Math.random()) * 400 + 50; // Orbiting radius from center
			const x = centerX + radius * Math.cos(angle);
			const y = centerY + radius * Math.sin(angle);
			const particle = new Particle(
				x,
				y,
				INITIAL_PARTICLE_MASS,
				"particle",
				this
			);
			const speed = 10 / Math.sqrt(radius);
			particle.vx = -speed * Math.sin(angle);
			particle.vy = speed * Math.cos(angle);
			this.particles.push(particle);
		}
	}
	randomizeCluster(
		count,
		clusterRadius = 50,
		initialVelocity = INITAL_VELOCITY
	) {
		const centerX = this.width / 2;
		const centerY = this.height / 2;

		for (let i = 0; i < count; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const radius = Math.sqrt(Math.random()) * clusterRadius;
			const x = centerX + radius * Math.cos(angle);
			const y = centerY + radius * Math.sin(angle);
			const particle = new Particle(
				x,
				y,
				INITIAL_PARTICLE_MASS,
				"particle",
				this
			);
			particle.vx = (Math.random() - 0.5) * initialVelocity;
			particle.vy = (Math.random() - 0.5) * initialVelocity;
			this.particles.push(particle);
		}
	}
	randomizeExpandingSphere(
		count,
		sphereRadius = 200,
		expansionSpeed = INITAL_VELOCITY
	) {
		const centerX = this.width / 2;
		const centerY = this.height / 2;

		for (let i = 0; i < count; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const elevation = (Math.random() - 0.5) * Math.PI; // elevation from the horizontal plane
			const radius = Math.sqrt(Math.random()) * sphereRadius;
			const x = centerX + radius * Math.sin(elevation) * Math.cos(angle);
			const y = centerY + radius * Math.sin(elevation) * Math.sin(angle);
			const z = radius * Math.cos(elevation); // If you use a 3D representation
			const particle = new Particle(
				x,
				y,
				INITIAL_PARTICLE_MASS,
				"particle",
				this
			);
			particle.vx = expansionSpeed * Math.sin(elevation) * Math.cos(angle);
			particle.vy = expansionSpeed * Math.sin(elevation) * Math.sin(angle);
			this.particles.push(particle);
		}
	}
	randomizeGrid(count) {
		const gridSpacing = Math.sqrt((this.width * this.height) / count);
		let id = 0;

		for (let x = gridSpacing / 2; x < this.width; x += gridSpacing) {
			for (let y = gridSpacing / 2; y < this.height; y += gridSpacing) {
				const particle = new Particle(
					x,
					y,
					INITIAL_PARTICLE_MASS,
					"particle",
					this
				);
				particle.vx = 0;
				particle.vy = 0;
				this.particles.push(particle);
				id++;
				if (id >= count) return; // Stop when we reach the desired count
			}
		}
	}
	randomizeTrinaryStarSystem(count, starMass = 1e8, separation = 200) {
		const centerX = this.width / 2;
		const centerY = this.height / 2;

		// Place three large central masses
		const star1 = new Particle(
			centerX,
			centerY - separation,
			starMass,
			"sun",
			this
		);
		const star2 = new Particle(
			centerX - separation * Math.cos(Math.PI / 6),
			centerY + separation * Math.sin(Math.PI / 6),
			starMass,
			"sun",
			this
		);
		const star3 = new Particle(
			centerX + separation * Math.cos(Math.PI / 6),
			centerY + separation * Math.sin(Math.PI / 6),
			starMass,
			"sun",
			this
		);
		this.sunCount = 3;
		// Setting initial velocities for a stable or interesting orbit if possible
		star1.vx = 1.5;
		star1.vy = 0;
		star2.vx = -0.75;
		star2.vy = -1.3;
		star3.vx = -0.75;
		star3.vy = 1.3;

		this.particles.push(star1, star2, star3);

		// Add smaller particles orbiting around the trinary stars
		for (let i = 0; i < count; i++) {
			const angle = Math.random() * 2 * Math.PI;
			const radius = Math.sqrt(Math.random()) * 300 + 100; // Adjust radius for visual appeal
			const x = centerX + radius * Math.cos(angle);
			const y = centerY + radius * Math.sin(angle);
			const particle = new Particle(
				x,
				y,
				INITIAL_PARTICLE_MASS,
				"particle",
				this
			);
			const speed = 8 / Math.sqrt(radius); // Adjust speed to manage orbits
			particle.vx = -speed * Math.sin(angle);
			particle.vy = speed * Math.cos(angle);
			this.particles.push(particle);
		}
	}
}

class Game {
	constructor(canvasId) {
		this.canvas = document.getElementById(canvasId);
		this.ctx = this.canvas.getContext("2d");

		// Initialize Simulation with canvas dimensions
		this.simulation = new Simulation(this.canvas.width, this.canvas.height);
		this.lastFrameTimeMs = 0;
		this.lastFpsUpdate = 0; // Time at which the last FPS update was performed
		this.framesThisSecond = 0; // Frames rendered in the current second
		this.fps = 0; // The FPS value

		this.paused = false;

		this.scale = 1;
		this.zoomFactor = 1.1; //zoom multiplier

		this.pan = { x: 0, y: 0 };
		this.dragging = false;
		this.lastMouse = { x: 0, y: 0 };

		this.initUIControls();
		this.initMouseControls();
	}
	zoomIn() {
		this.scale *= this.zoomFactor;
		this.updateTransform();
	}

	zoomOut() {
		this.scale /= this.zoomFactor;
		this.updateTransform();
	}

	updateTransform(mouseX, mouseY, isZoomIn) {
		const rect = this.canvas.getBoundingClientRect();
		const x = (mouseX - rect.left) / this.scale; // Adjust mouse X considering current scale
		const y = (mouseY - rect.top) / this.scale; // Adjust mouse Y considering current scale

		this.ctx.translate(x, y); // Move context to mouse position

		if (isZoomIn) {
			this.scale *= this.zoomFactor;
			this.ctx.scale(this.zoomFactor, this.zoomFactor);
		} else {
			this.scale /= this.zoomFactor;
			this.ctx.scale(1 / this.zoomFactor, 1 / this.zoomFactor);
		}

		//this.ctx.scale(this.zoomFactor, this.zoomFactor);  // Apply scaling
		this.ctx.translate(-x, -y); // Reset context position

		this.render(); // Redraw the canvas with the new transformations
	}

	updateParticleCountDisplay() {
		const totalParticles = this.simulation.particles.length;
		const planetCount = this.simulation.planetCount;
		const sunCount = this.simulation.sunCount;
		const blackHoleCount = this.simulation.blackHoleCount;

		document.getElementById("particleCount").textContent = `Particles: ${
			totalParticles - sunCount - planetCount - blackHoleCount
		}`;
		document.getElementById(
			"planetCount"
		).textContent = `Planets: ${planetCount}`;
		document.getElementById("starCount").textContent = `Stars: ${sunCount}`;
		document.getElementById(
			"blackholeCount"
		).textContent = `Black Holes: ${blackHoleCount}`;
	}

	updateMaxVelocityDisplay() {
		const maxV = this.simulation.currentMaxVelocity;

		document.getElementById(
			"maxVelocity"
		).textContent = `Max Velocity: ${maxV}`;
	}

	initUIControls() {
		document
			.getElementById("pauseButton")
			.addEventListener("click", () => this.togglePause());
		document
			.getElementById("resetZoomButton")
			.addEventListener("click", () => this.resetZoom());

		document.getElementById("gScaler").addEventListener("change", function () {
			G_SCALER = parseFloat(this.value);
			G = 6.6743e-11 * G_SCALER; // Update the actual gravitational constant used in simulation
			document.getElementById("gConstant").value = G.toFixed(10); // Update display
		});

		document.getElementById("eta").addEventListener("change", function () {
			ETA = parseFloat(this.value);
		});

		document.getElementById("dt").addEventListener("change", function () {
			DT = parseFloat(this.value);
		});

		document.getElementById("maxDepth").addEventListener("change", function () {
			MAX_DEPTH = parseFloat(this.value);
		});

		document.getElementById("theta").addEventListener("change", function () {
			THETA = parseFloat(this.value);
		});

		document
			.getElementById("planetThreshold")
			.addEventListener("change", function () {
				PLANET_ACCRETION_THRESHOLD = parseFloat(this.value);
			});
		document
			.getElementById("sunThreshold")
			.addEventListener("change", function () {
				SUN_ACCRETION_THRESHOLD = parseFloat(this.value);
			});
		document
			.getElementById("supernovaAge")
			.addEventListener("change", function () {
				SUN_TO_SUPERNOVA_AGE = parseFloat(this.value);
			});
		document
			.getElementById("accretionChance")
			.addEventListener("change", function () {
				ACCRETION_CHANCE = parseFloat(this.value);
			});
		document
			.getElementById("supernovaChance")
			.addEventListener("change", function () {
				SUPERNOVA_CHANCE = parseFloat(this.value);
			});

		document
			.getElementById("maxExplosion")
			.addEventListener("change", function () {
				MAX_EXPLOSION_SPEED = parseFloat(this.value);
			});

		document
			.getElementById("blackholeChance")
			.addEventListener("change", function () {
				BLACKHOLE_CHANCE = parseFloat(this.value);
			});

		document
			.getElementById("blackholeAccretion")
			.addEventListener("change", function () {
				BLACKHOLE_ACCRETION_CHANCE = parseFloat(this.value);
			});

		document
			.getElementById("blackholeHorizon")
			.addEventListener("change", function () {
				BLACKHOLE_EVENT_HORIZON = parseFloat(this.value);
			});

		document.getElementById("maxMass").addEventListener("change", function () {
			MAX_MASS = parseFloat(this.value);
		});

		document
			.getElementById("supernovaParticleGen")
			.addEventListener("change", function () {
				SUPERNOVA_PARTICLE_GEN_MULTIPLIER = parseFloat(this.value);
			});

		//inital conditions
		document
			.getElementById("initialParticleCount")
			.addEventListener("change", function () {
				INITIAL_PARTICLE_COUNT = parseFloat(this.value);
			});
		document
			.getElementById("initialParticleMass")
			.addEventListener("change", function () {
				INITIAL_PARTICLE_MASS = parseFloat(this.value);
			});
		document
			.getElementById("initialVelocity")
			.addEventListener("change", function () {
				INITAL_VELOCITY = parseFloat(this.value);
			});
		document
			.getElementById("radiusRotation")
			.addEventListener("change", function () {
				RADIUS_ROTATION = parseFloat(this.value);
			});

		//rendering/visuals
		document
			.getElementById("renderQuadtree")
			.addEventListener("change", function () {
				RENDER_QUADTREE = this.checked; // Directly use the checked property to update the flag
			});

		document
			.getElementById("renderCenterMass")
			.addEventListener("change", function () {
				RENDER_CENTER_OF_MASS = this.checked;
			});
		document
			.getElementById("renderTrail")
			.addEventListener("change", function () {
				RENDER_TRAIL = this.checked;
			});
		document
			.getElementById("particleRadius")
			.addEventListener("change", function () {
				PARTICLE_RADIUS = parseFloat(this.value);
			});

		document
			.getElementById("planetRadius")
			.addEventListener("change", function () {
				PLANET_RADIUS = parseFloat(this.value);
			});

		document
			.getElementById("sunRadius")
			.addEventListener("change", function () {
				SUN_RADIUS = parseFloat(this.value);
			});

		document
			.getElementById("blackholeRadius")
			.addEventListener("change", function () {
				BLACKHOLE_RADIUS = parseFloat(this.value);
			});

		document.getElementById("maxTrail").addEventListener("change", function () {
			MAX_TRAIL = parseFloat(this.value);
		});

		document
			.getElementById("trailAlpha")
			.addEventListener("change", function () {
				TRAIL_ALPHA = parseFloat(this.value);
			});

		//color pickers
		document
			.getElementById("particleColorSlow")
			.addEventListener("change", function () {
				PARTICLE_COLOR_SLOW = hexToRgb(this.value); // Convert hex to RGB array
			});

		document
			.getElementById("particleColorFast")
			.addEventListener("change", function () {
				PARTICLE_COLOR_FAST = hexToRgb(this.value);
			});

		document
			.getElementById("planetColor")
			.addEventListener("change", function () {
				PLANET_COLOR = this.value; // Directly use the hex value
			});

		document.getElementById("sunColor").addEventListener("change", function () {
			SUN_COLOR = this.value;
		});

		document
			.getElementById("blackholeColor")
			.addEventListener("change", function () {
				BLACKHOLE_COLOR = this.value;
			});

		document
			.getElementById("helpButton")
			.addEventListener("click", function () {
				var helpSection = document.getElementById("helpSection");
				if (helpSection.style.display === "none") {
					helpSection.style.display = "block";
				} else {
					helpSection.style.display = "none";
				}
			});
	}

	resetZoom() {
		this.scale = 1; // Reset scale to initial value
		this.pan = { x: 0, y: 0 }; // Reset panning to initial values
		this.render(); // Re-render the canvas with reset transformations
	}
	initMouseControls() {
		this.canvas.addEventListener("wheel", (event) => {
			event.preventDefault();
			this.handleZoom(event.deltaY, event.clientX, event.clientY);
		});

		this.canvas.addEventListener("mousedown", (event) => {
			if (event.button === 1) {
				// Middle mouse button for panning
				this.dragging = true;
				this.lastMouse.x = event.clientX;
				this.lastMouse.y = event.clientY;
			} else if (event.button === 0) {
				// Left mouse button for drawing
				this.startDrawing(event);
			}
		});

		this.canvas.addEventListener("mousemove", (event) => {
			if (this.dragging) {
				const dx = (event.clientX - this.lastMouse.x) / this.scale;
				const dy = (event.clientY - this.lastMouse.y) / this.scale;
				this.pan.x += dx;
				this.pan.y += dy;
				this.lastMouse.x = event.clientX;
				this.lastMouse.y = event.clientY;
				this.render();
			} else if (event.buttons === 1) {
				// Check if left mouse button is held down for drawing
				this.drawParticleOnMove(event);
			}
		});

		this.canvas.addEventListener("mouseup", (event) => {
			if (event.button === 1) {
				// Middle mouse button
				this.dragging = false;
			}
		});

		this.canvas.addEventListener("mouseout", () => {
			this.dragging = false;
		});
	}

	handleZoom(deltaY, mouseX, mouseY) {
		const rect = this.canvas.getBoundingClientRect();
		const x = mouseX - rect.left - this.pan.x; // Mouse X position relative to the canvas, including current pan
		const y = mouseY - rect.top - this.pan.y; // Mouse Y position relative to the canvas, including current pan

		// Calculate the new scale based on the direction of the wheel scroll
		const oldScale = this.scale;
		if (deltaY < 0) {
			this.scale *= this.zoomFactor;
		} else {
			this.scale /= this.zoomFactor;
		}

		// Adjust pan to keep the mouse point stationary
		this.pan.x -= x * (this.scale / oldScale - 1);
		this.pan.y -= y * (this.scale / oldScale - 1);

		this.render(); // Redraw the canvas with the new transformations
	}

	startDrawing(event) {
		const rect = this.canvas.getBoundingClientRect();
		// Calculate the correct coordinates with current pan and scale
		const x = (event.clientX - rect.left - this.pan.x) / this.scale;
		const y = (event.clientY - rect.top - this.pan.y) / this.scale;
		const type = document.getElementById("objectType").value;
		this.createParticle(x, y, type);
	}

	drawParticleOnMove(event) {
		if (event.buttons === 1) {
			// Left mouse button pressed
			const rect = this.canvas.getBoundingClientRect();
			// Calculate the correct coordinates with current pan and scale
			const x = (event.clientX - rect.left - this.pan.x) / this.scale;
			const y = (event.clientY - rect.top - this.pan.y) / this.scale;
			const type = document.getElementById("objectType").value;
			this.createParticle(x, y, type);
		}
	}

	createParticle(x, y, type) {
		let mass = INITIAL_PARTICLE_MASS;
		let accretedParticles = [];
		switch (type) {
			case "planet":
				mass *= PLANET_ACCRETION_THRESHOLD;
				this.simulation.planetCount++;
				break;
			case "sun":
				mass *= SUN_ACCRETION_THRESHOLD;
				this.simulation.sunCount++;
				for (let i = 0; i < SUN_ACCRETION_THRESHOLD; i++) {
					accretedParticles.push(
						new Particle(x, y, INITIAL_PARTICLE_MASS, "particle")
					);
				}
				break;
			case "blackhole":
				mass = MAX_MASS; // Use predefined max mass for black holes
				this.simulation.blackHoleCount++;
				break;
			default:
				// This will handle the "particle" case
				break;
		}
		const particle = new Particle(x, y, mass, type);
		particle.accretedParticles = accretedParticles;
		particle.accretionCount = accretedParticles.length; // Set the accretion count
		this.simulation.particles.push(particle);
		this.updateParticleCountDisplay(); // Update UI to reflect the new counts

		this.render();
	}

	togglePause() {
		this.paused = !this.paused;
		if (!this.paused) {
			requestAnimationFrame(this.loop.bind(this));
		}
	}

	start() {
		//this.simulation.randomizeParticles(INITIAL_PARTICLE_COUNT); // Example: start with 50 particles
		this.simulation.randomizeRotatingDisk(
			INITIAL_PARTICLE_COUNT,
			RADIUS_ROTATION
		);

		requestAnimationFrame(this.loop.bind(this));
	}

	loop(timestamp) {
		if (this.paused) {
			this.lastFrameTimeMs = timestamp; // Ensure smooth animation after unpausing
			return;
		}

		const delta = timestamp - this.lastFrameTimeMs;
		this.lastFrameTimeMs = timestamp;
		this.framesThisSecond++;

		if (timestamp > this.lastFpsUpdate + 1000) {
			// Update every second
			this.fps = this.framesThisSecond;
			this.framesThisSecond = 0;
			this.lastFpsUpdate = timestamp;
			document.getElementById("fpsCount").textContent = `FPS: ${this.fps}`;
		}

		if (!this.lastFrameTimeMs || delta >= 1000 / 60) {
			this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
			this.updateParticleCountDisplay();
			this.updateMaxVelocityDisplay();
			this.simulation.update();
			this.render();
		}
		requestAnimationFrame(this.loop.bind(this));
	}

	render() {
		this.ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); // Clear the canvas
		this.ctx.translate(this.pan.x, this.pan.y); // Apply panning
		this.ctx.scale(this.scale, this.scale); // Apply scaling

		if (RENDER_QUADTREE) this.drawQuadtree(this.simulation.rootNode);

		if (RENDER_CENTER_OF_MASS) this.drawSystemCenterOfMass();

		this.simulation.particles.forEach((particle) => {
			this.drawParticle(particle);
		});
	}

	drawParticle(particle) {
		if (RENDER_TRAIL) {
			this.ctx.beginPath();
			let previousPos = { x: particle.x, y: particle.y };

			// Iterate over the trail positions in reverse to fade out
			particle.trail.forEach((pos, index) => {
				this.ctx.strokeStyle = `rgba(255, 255, 255, ${Math.max(
					0,
					TRAIL_ALPHA - index / (MAX_TRAIL / 2)
				)})`; // Fading faster
				this.ctx.lineWidth = 1; // You can adjust line width for better visibility
				this.ctx.beginPath();
				this.ctx.moveTo(previousPos.x, previousPos.y);
				this.ctx.lineTo(pos.x, pos.y);
				this.ctx.stroke();
				previousPos = pos;
			});
		}

		this.ctx.beginPath();

		// Determine the size based on the type of particle
		let radius = PARTICLE_RADIUS; // Default radius for basic particles
		if (particle.type === "planet") {
			radius = PLANET_RADIUS; // Larger radius for planets
			this.ctx.fillStyle = PLANET_COLOR || "lime";
		} else if (particle.type === "sun") {
			radius = SUN_RADIUS; // Even larger radius for suns
			this.ctx.fillStyle = SUN_COLOR || "yellow";
		} else if (particle.type === "blackhole") {
			radius = BLACKHOLE_RADIUS; // Black holes could be visualized distinctively if desired
			this.ctx.fillStyle = BLACKHOLE_COLOR || "maroon";
		} else {
			// Use a dynamic color based on the particle's velocity for other particles
			const velocityMagnitude = Math.sqrt(particle.vx ** 2 + particle.vy ** 2);
			this.ctx.fillStyle = this.getVelocityColor(velocityMagnitude);
		}

		// Draw the circle
		this.ctx.arc(
			particle.x,
			particle.y,
			radius, // Use the dynamically set radius
			0,
			2 * Math.PI // Draw a complete circle
		);
		this.ctx.fill();
	}

	// Linearly interpolates between two colors
	interpolateColor(color1, color2, factor) {
		let result = color1.slice();
		for (let i = 0; i < 3; i++) {
			result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
		}
		return "rgb(" + result.join(",") + ")";
	}

	// Method to determine the color based on velocity magnitude
	getVelocityColor(velocity) {
		// Define the velocity range for the simulation
		const minVelocity = 0; // Minimum velocity
		//const maxVelocity = 40; // Maximum expected velocity, adjust as needed
		const maxVelocity = this.simulation.currentMaxVelocity;

		// Colors defined in RGB for the start and end of the gradient
		const startColor = PARTICLE_COLOR_SLOW; // Light blue
		const endColor = PARTICLE_COLOR_FAST; // dark blue

		// Calculate interpolation factor (0 to 1) based on velocity
		const factor = (velocity - minVelocity) / (maxVelocity - minVelocity);

		// Ensure the factor is clamped between 0 and 1
		const clampedFactor = Math.min(Math.max(factor, 0), 1);

		// Interpolate between the start and end colors based on the velocity factor
		return this.interpolateColor(startColor, endColor, clampedFactor);
	}

	drawSystemCenterOfMass() {
		const com = this.simulation.rootNode.centerOfMass;
		this.ctx.fillStyle = "pink"; // Use a distinctive color for the system's center of mass
		this.ctx.beginPath();
		this.ctx.arc(com.x, com.y, 5, 0, 2 * Math.PI); // A slightly larger dot
		this.ctx.fill();
	}
	drawQuadtree(node) {
		if (!node) {
			return;
		}

		// Draw the boundary of the current quadtree node
		this.ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"; // Light grey color for the boundary
		this.ctx.strokeRect(
			node.boundary.x,
			node.boundary.y,
			node.boundary.width,
			node.boundary.height
		);

		// Draw the center of mass for the current node
		if (node.totalMass > 0) {
			this.ctx.fillStyle = "red"; // Red color for the center of mass
			this.ctx.beginPath();
			this.ctx.arc(node.centerOfMass.x, node.centerOfMass.y, 3, 0, 2 * Math.PI);
			this.ctx.fill();
		}

		// Recursively draw children
		if (node.isDivided) {
			node.children.forEach((child) => this.drawQuadtree(child));
		}
	}

	// Additional methods for user interaction, particle creation, etc.
}

document.addEventListener("DOMContentLoaded", (event) => {
	const game = new Game("simulationCanvas");
	game.start();

	window.updateCanvasSize = function () {
		const width = document.getElementById("widthInput").value || 800;
		const height = document.getElementById("heightInput").value || 600;
		const canvas = document.getElementById("simulationCanvas");
		canvas.width = width;
		canvas.height = height;
	};

	window.restartSimulation = function () {
		// Retrieve current canvas dimensions
		const canvas = document.getElementById("simulationCanvas");
		const width = canvas.width;
		const height = canvas.height;

		// Reset the simulation with the current dimensions
		game.simulation = new Simulation(width, height);

		// Initialize simulation based on selected initial condition
		const initialCondition = document.getElementById("initialCondition").value;
		switch (initialCondition) {
			case "rotatingDisk":
				game.simulation.randomizeRotatingDisk(
					INITIAL_PARTICLE_COUNT,
					RADIUS_ROTATION,
					INITAL_VELOCITY
				);
				break;
			case "randomDistribution":
				game.simulation.randomizeParticles(INITIAL_PARTICLE_COUNT);
				break;
			case "binary":
				game.simulation.randomizeBinaryStarSystem(
					INITIAL_PARTICLE_COUNT,
					INITIAL_PARTICLE_MASS * SUN_ACCRETION_THRESHOLD,
					RADIUS_ROTATION
				);
				break;
			case "trinary":
				game.simulation.randomizeTrinaryStarSystem(
					INITIAL_PARTICLE_COUNT,
					INITIAL_PARTICLE_MASS * SUN_ACCRETION_THRESHOLD,
					RADIUS_ROTATION
				);
				break;
			case "cluster":
				game.simulation.randomizeCluster(
					INITIAL_PARTICLE_COUNT,
					RADIUS_ROTATION,
					INITAL_VELOCITY
				);
			case "expandingSphere":
				game.simulation.randomizeExpandingSphere(
					INITIAL_PARTICLE_COUNT,
					RADIUS_ROTATION,
					INITAL_VELOCITY
				);
				break;

			case "grid":
				game.simulation.randomizeGrid(INITIAL_PARTICLE_COUNT);
				break;
			case "empty":
				game.simulation.particles = [];
			// Add cases for other initial conditions as needed
		}

		if (!game.paused) {
			requestAnimationFrame(game.loop.bind(game));
		}
	};
});

function hexToRgb(hex) {
	var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return result
		? [
				parseInt(result[1], 16),
				parseInt(result[2], 16),
				parseInt(result[3], 16),
		  ]
		: null;
}
