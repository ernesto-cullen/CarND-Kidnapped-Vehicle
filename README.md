# Overview
This repository contains the code for the Particle Filters project in Udacity's Self-Driving car nanodegree, term 2.

The project implements a 2 dimensional particle filter.


## Project Introduction
Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

This project implements a 2 dimensional particle filter in C++. Given a map and some initial localization information (analogous to what a GPS would provide), the filter will get observation and control data on discrete time steps. The filter will adjust its estimation of the position and orientation of the vehicle using a swarm of particles and the landmarks on the map.


## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./particle_filter

Alternatively some scripts have been included to streamline this process, these can be leveraged by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh


# Implementing the Particle Filter
All the code for the particle filter is encapsulated in the class `ParticleFilter`, implemented in files `particle_filter.cpp` and `particle_filter.h`. This class will be used from the main program in `main.cpp` which will also communicate with the simulator.

## Inputs to the Particle Filter
The input to the particle filter is the map with 42 landmarks data, in `data` directory. This file will be read and stored in memory by the main program.


#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

### Observations on each step come from the simulator

> * Map data provided by 3D Mapping Solutions GmbH.

## The code
The particle filter begins by initializing the particles in method `ParticleFilter::init`. Here you can define the number of particles, which are then initialized with random positions and orientations. The initialization includes some random gaussian noise for each parameter: the standard deviation of the noise is sent from the main program.

```C++
void ParticleFilter::init(double x, double y, double theta, double std[]) {
    std::default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 50;
	for(int i=0; i<num_particles; i++) {
		Particle p = Particle();
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0d;
		particles.push_back(p);
		weights.push_back(1.0d);
	}
	is_initialized = true;
}
```

The main program calls this initialization and then starts the standard cycle of the filter: prediction - measure - update.
In the prediction step, we are given the linear velocity and the yaw rate of the vehicle; we incorporate these in all the particles adding noise to account for inexact measurement:

```C++
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    std::default_random_engine gen;
	for (Particle& p:particles) {
		if (fabs(yaw_rate) <0.0001) {
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
			p.theta = p.theta;
		} else {
			p.x = p.x + velocity/yaw_rate * (sin(p.theta+yaw_rate*delta_t) - sin(p.theta));
			p.y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta+yaw_rate*delta_t));
			p.theta = p.theta + yaw_rate * delta_t;
		}
        //add noise
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_theta(p.theta, std_pos[2]);
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}
```

next, we are given a set of observations by the simulator. We incorporate these by associating each particle with the nearest landmark in the map. Each particle then gets a _weight_ indicating how correct this particle is in its measurement: the more similar the observation to the correct position of the landmark the bigger the weight.

```c++
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    for (int ind=0; ind<num_particles; ind++) {
        Particle p{particles[ind].id, particles[ind].x, particles[ind].y, particles[ind].theta, 1.0};
        vector<int> assoc;
        vector<double> sense_x;
        vector<double> sense_y;

        //get a list of landmarks within sensor range of the particle
        std::vector<LandmarkObs> lmkInRange;
        for (auto lmk : map_landmarks.landmark_list) {
            if (dist(p.x,p.y,lmk.x_f,lmk.y_f) <= sensor_range) {
                lmkInRange.push_back(LandmarkObs{lmk.id_i, lmk.x_f, lmk.y_f});
            }
        }

        //update particle weight
        for (int i=0; i<observations.size(); i++) {
            //observation made from the car
            LandmarkObs obs;
            //transform observation from car frame to map frame (for this particle)
            obs.x = p.x + cos(p.theta)*observations[i].x - sin(p.theta)*observations[i].y;
            obs.y = p.y + sin(p.theta)*observations[i].x + cos(p.theta)*observations[i].y;
            obs.id = observations[i].id;

            //look for the landmark closer to this obs
            double min_dist = 0.0;
            LandmarkObs min_pred{};
            bool first_dist = true;
            for (int k=0; k<lmkInRange.size(); k++) {
                LandmarkObs pred = lmkInRange[k];
                double distance = dist(obs.x,obs.y, pred.x,pred.y);
                if (first_dist || distance < min_dist) {
                    min_dist = distance;
                    min_pred = pred;
                    first_dist = false;
                }
            }

            //calculate weight for this observation
            double obs_weight = ( 1/(2*M_PI*std_x*std_y)) *
                    exp( -(pow(obs.x-min_pred.x,2)/(2*pow(std_x, 2)) + pow(obs.y-min_pred.y,2)/(2*pow(std_y, 2))) );

            //update particle's weight
            p.weight *= obs_weight;
            assoc.push_back(min_pred.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }
        particles[ind] = SetAssociations(p, assoc, sense_x, sense_y);
        weights[ind] = p.weight; //weights list to use in the resample procedure
    }
}
```
*Note*: the actual code allows the printing of the particles, observations and weights for debugging purposes. This code is executed if `ParticleFilter::debug` is true. It is not shown here.

After the weights computation, the particles are _resampled_ randomly but taking into account their weight: particles with bigger weight are selected more frequently. After this resampling, we are left with the set of particles that are more probably correct in their estimation of the vehicle's position and orientation (with the best ones repeated):

```c++
void ParticleFilter::resample() {
    default_random_engine gen;
    discrete_distribution<int> dist_p(weights.begin(), weights.end());

    //create a new vector with particles resampled by weight
    vector<Particle> resampled_particles;
    int index = 0;
    for (int i=0; i<num_particles; i++) {
        index = dist_p(gen);
        resampled_particles.push_back(particles[index]);
        resampled_particles[i].id = i;
    }
    particles = resampled_particles;
}
```

And that's it: the main program sends the particles to the simulator which draw the best particle's pose (the one with the bigger weight) and the recognized landmarks. It also computes and display the mean error in each dimension.


## Conclusion
The particle filter is very simple in its implementation, once you get the idea. It does not require complex mathematical formulas and it can be made very precise by using more particles.

I struggled some time with implementation details, like using an already updated coordinate to compute the other or making changes on copies of objects instead of the original ones; that's why I had to add all the debug code. Using few particles (say 5) and turning on debugging by setting `ParticleFilter::debug` to `true`, you can see the computations as they are performed and spot difficult problems.
