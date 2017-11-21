/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <random>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
    std::mt19937 gen;
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

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::mt19937 gen;
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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: not used. I rolled up this code in the updateWeights methods to optimize
    for (int i=0; i<observations.size(); i++) {
        LandmarkObs p_obs = observations[i];
        double minDistance = 0.0;
        int minId = -1;
        bool firstDistance = true;
        for (int j=0; j<predicted.size(); j++) {
            LandmarkObs p_pred = predicted[j];
            double distance = dist(p_obs.x,p_obs.y, p_pred.x,p_pred.y);
            if (firstDistance || distance < minDistance) {
                minDistance = distance;
                minId = p_pred.id;
            }
            firstDistance = false;
        }
        p_obs.id = minId;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    double std_x = std_landmark[0];
    double std_y = std_landmark[1];

    if (debug) {
        cout<<"Particles before update weights:"<<endl;
        printParticles();

        cout << "Updated weights:" << endl;
    }

    for (int ind=0; ind<num_particles; ind++) {
        Particle p{particles[ind].id, particles[ind].x, particles[ind].y, particles[ind].theta, 1.0};

        if (debug) {
            cout<<"Particle: "<<p.id<<" ("<<p.x<<","<<p.y<<")"<<endl;
        }

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

        if (debug) {
            cout<<"There are "<<lmkInRange.size()<<" landmarks in sensor range"<<endl;
            for (auto lmk : lmkInRange) {
                cout<<"lmk"<<lmk.id<<": ("<<lmk.x<<","<<lmk.y<<")"<<endl;
            }
            cout<<endl;
        }
        if (debug) {
            cout<<"Observations: "<<observations.size()<<endl;
            for (auto lmk : observations) {
                cout<<"obs"<<lmk.id<<": ("<<lmk.x<<","<<lmk.y<<")"<<endl;
            }
            cout<<endl;
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
            if (debug) {
                cout<<"obs"<<obs.id<<" in map coords: ("<<obs.x<<","<<obs.y<<")"<<endl;
                cout<<"  closest landmark: lmk"<<min_pred.id<<" ("<<min_pred.x<<","<<min_pred.y<<") dist: "<<min_dist<<endl;
            }
            //calculate weight for this observation
            double obs_weight = ( 1/(2*M_PI*std_x*std_y)) *
                    exp( -(pow(obs.x-min_pred.x,2)/(2*pow(std_x, 2)) + pow(obs.y-min_pred.y,2)/(2*pow(std_y, 2))) );
            if (debug) cout<<"obs_weight: "<<obs_weight<<endl;
            //update particle's weight
            p.weight *= obs_weight;
            assoc.push_back(min_pred.id);
            sense_x.push_back(obs.x);
            sense_y.push_back(obs.y);
        }
        particles[ind] = SetAssociations(p, assoc, sense_x, sense_y);
        weights[ind] = p.weight; //weights list to use in the resample procedure
        if (debug) cout << "Final weight: "<<p.weight << endl;
    }
    if (debug) cout << endl << "----" << endl;
}

void ParticleFilter::resample() {
    default_random_engine gen;
    discrete_distribution<int> dist_p(weights.begin(), weights.end());

    if (debug) {
        cout << "weights before resampling" << endl;
        for (int i = 0; i < weights.size(); i++) {
            cout << i << ": " << weights[i] << endl;
        }
        cout << endl;

        cout << "particles before resample" << endl;
        printParticles();
    }
    //create a new vector with particles resampled by weight
    vector<Particle> resampled_particles;
    int index = 0;
    for (int i=0; i<num_particles; i++) {
        index = dist_p(gen);
        if (debug) cout<<"index: "<<index<<endl;
        resampled_particles.push_back(particles[index]);
        resampled_particles[i].id = i;
    }
    particles = resampled_particles;
    if (debug) {
        cout << "particles after resample" << endl;
        printParticles();

    }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::printParticles() {
    for (int i = 0; i < num_particles; i++) {
        printParticle(particles[i], i);
    }
    cout << endl;
}

void ParticleFilter::printParticle(Particle p, int i) {
    cout << "P" << i << ": (" << p.x << "," << p.y << "), theta: " << p.theta
         << ", w: " << p.weight << endl;
}
