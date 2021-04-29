extern crate karmarkar as kk;
extern crate nalgebra as na;
use kk::karmarkar;
use na::{DMatrix, DVector};

pub fn main() {
    let c = DVector::<f64>::from_vec(vec![-1.0, -1.0]);
    let amat = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 1.0, 1.0, -1.0]);
    let b = DVector::<f64>::from_vec(vec![0.5, 1.0]);
    let x = DVector::<f64>::from_vec(vec![-2.0, -2.0]);
    let ans = karmarkar(&c, &amat, &b, &x, 0.5, 1.0e-3, 30);
    println!("{:?}", ans);
}
