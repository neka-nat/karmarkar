extern crate karmarkar as kk;
extern crate nalgebra as na;
use kk::karmarkar;
use na::{MatrixN, VectorN, U2};

pub fn main() {
    // max c.T * x, subj A * x < b
    let c = VectorN::<f64, U2>::from_vec(vec![-1.0, -1.0]);
    let amat = MatrixN::<f64, U2>::from_vec(vec![1.0, 1.0, 1.0, -1.0]);
    let b = VectorN::<f64, U2>::from_vec(vec![0.5, 1.0]);
    let x = VectorN::<f64, U2>::from_vec(vec![-2.0, -2.0]);
    let ans = karmarkar(&c, &amat, &b, &x, 0.5, 1.0e-3, 30);
    println!("{:?}", ans);
}
