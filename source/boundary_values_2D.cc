#include "boundary_values_2D.h"
#include <cmath>
#define PI 3.14159265358979323846

using namespace dealii;

// - - - - -  public functions - - - - -
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim>  &p,
                            const unsigned int component) const
{
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

  Vector<double> values(2);
  BoundaryValues<dim>::vector_value (p, values);

  if (component == 0)
      return values(0);
  if (component == 1)
      return values(1);
  return 0;
}

template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
    if(derivative)
        BoundaryValues<dim>::get_values_dt(p, values);
    else
    {
        if (color == 3) {
            get_heartdelta(p, values, heartstep);
        }
        else
            BoundaryValues<dim>::get_values(p, values);
    }
}



// - - - - -  private functions - - - - -

template <int dim>
void
BoundaryValues<dim>::transform_to_polar_coord (const Point<3> &p,
                                               double rot,
                                               double &angle, 
                                               double &height) const
{
  // convert point to polar coordinates
  double x   = p[0],
         y   = p[1],
         z   = p[2],
         phi = atan2(y,z); // returns angle in the range from -Pi to Pi

  // need to rotate to match heart_fe
  phi = phi+rot;

  // angle needs to be in the range from 0 to 2Pi
  phi = (phi < 0) ? phi+2*PI : phi;
  
  angle = phi;
  height = x;
}

template <int dim>
void
BoundaryValues<dim>::swap_coord (Point<3> &p) const
{
  // swap x and z coordinate
  double tmp = p(0);
  //std::cout << "Hallo error1" << std::endl;
  p(0) = p(2);
  p(2) = tmp;
}

template <int dim>
void
BoundaryValues<dim>::get_heartdelta (const Point<dim> &p,
                                     Vector<double>   &values,
                                     int heartstep) const
{
  //onst Point<3> p (point(1), 0, point(0)); //?
  if (color == 3)         //////////////////////////////// top face 
  {
      // convert to polar coordinates and rotate 45 degrees
      double phi, h, rot = -PI/4;
      Point<3> artificial_p (p(1), p(0), 0);
      transform_to_polar_coord(artificial_p, rot, phi, h);

      // transform back to cartesian
      double x, y, r;
      x = p(0);
      y = 0;

      r = sqrt(x*x + y*y);
      x = r * cos(phi);
      y = r * sin(phi);
      
      // calc delta
      values(0) = y-p(0);
      values(1) = 0;
  }
  else if (color == 2)    //////////////////////////////// bottom face
  {
      double x, y;
      x = p(0);
      y = p(0);

      Point<2> two_dim_pnt (x, y);
      //std::cout << "(" << x << ", " << y << ")" << std::endl;
      //get heart boundary point
      Point<3> heart_p = heart.push_forward (two_dim_pnt, heartstep);
      swap_coord(heart_p);
      //std::cout << "Hallo error2" << std::endl;
      // calc delta
      values(0) = heart_p(1) - p(0);
      values(1) = heart_p(0) - p(1);
      //std::cout << "Hallo error3" << std::endl;
      //values(0) = 0;
      //values(1) = 0;
  }
  else if (color == 0 || color == 1)    //////////////////////////////// hull
  {
      // convert to polar coordinates and rotate -45 degrees
      double phi, h, rot = -PI/2;
      Point<3> artificial_p (p(1), p(0), 0);
      transform_to_polar_coord(artificial_p, rot, phi, h);
      Point<2> polar_pnt (phi, h);
      
      //get heart boundary point
      Point<3> heart_p = heart.push_forward (polar_pnt, heartstep);
      swap_coord(heart_p);

      // calc delta
      //values(0) = heart_p(0) - p(0);
      //values(1) = heart_p(1) - p(1);
      //values(2) = heart_p(2) - p(2);
      values(0) = heart_p(1) - p(0);
      values(1) = heart_p(0) - p(1);
  }
}

template <int dim>
void
BoundaryValues<dim>::get_values (const Point<dim> &p,
                                 Vector<double>   &values) const
{
    Vector<double> u_(dim);                                 // u_t-1
    Vector<double> u(dim);                                  // u_t
    Vector<double> delta_u(dim);                            // u_t - u_t-1
    double substep = (fmod (timestep, heartinterval)/dt + 1)
                     / (heartinterval / dt);

    BoundaryValues<dim>::get_heartdelta(p, u_, (heartstep-1 < 0) ? 0 : heartstep-1);
    BoundaryValues<dim>::get_heartdelta(p, u, heartstep);

    // calc delta_u
    delta_u = u;
    delta_u -= u_;
    // scale delta_u
    delta_u *= substep;

    u_ += delta_u;

    values(0) = u_(0);
    values(1) = u_(1);
    //values(2) = u_(2);
}



template <int dim>
void
BoundaryValues<dim>::get_values_dt (const Point<dim> &p,
                                    Vector<double>   &values) const
{
    Vector<double> u_(dim);       // u_t-1
    Vector<double> u(dim);        // u_t
    Vector<double> delta_u(dim);  // u_t - u_t-1
    //double substep = (fmod (timestep, heartinterval)/dt + 1)
    //                 / (heartinterval / dt);

    BoundaryValues<dim>::get_heartdelta(p, u_, (heartstep-1 < 0) ? 0 : heartstep-1);
    BoundaryValues<dim>::get_heartdelta(p, u, heartstep);

    // (u_t - u_t-1) / h
    delta_u = u;
    delta_u -= u_;
    delta_u /= heartinterval;
    // scale u_
    delta_u /= (heartinterval/dt);

    values(0) = delta_u(0);
    values(1) = delta_u(1);
    //values(2) = u_(2);
}

// Explicit instantiations
template class BoundaryValues<2>;