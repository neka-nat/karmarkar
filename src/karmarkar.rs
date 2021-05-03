extern crate nalgebra as na;
use na::{DMatrix, DVector};

pub fn karmarkar(
    c: &DVector<f64>,
    amat: &DMatrix<f64>,
    b: &DVector<f64>,
    x: &DVector<f64>,
    gamma: f64,
    eps: f64,
    nloop: i32,
) -> Result<DVector<f64>, &'static str> {
    let mut ans = x.clone();
    for _ in 0..nloop {
        let vk = b - amat * &ans;
        let mut vk2 = DVector::<f64>::zeros(vk.len());
        vk2.cmpy(1.0, &vk, &vk, 0.0);
        let ivk2 = DMatrix::<f64>::from_diagonal(&vk2)
            .try_inverse()
            .ok_or("Not found inverse.")?;
        let gmat = amat.transpose() * ivk2 * amat;
        let pgmat = gmat.pseudo_inverse(1.0e-9)?;
        let d = pgmat * c;
        if d.norm() < eps {
            break;
        }
        let hv = -amat * &d;
        if hv.amax() <= 0.0 {
            return Err("Unbounded!");
        }
        let sa = (0..hv.nrows())
            .filter_map(|i| {
                if hv[i] > 0.0 {
                    Some(vk[i] / hv[i])
                } else {
                    None
                }
            })
            .fold(0.0 / 0.0, |m, v| v.min(m));
        if sa.is_nan() {
            break;
        }
        let alpha = gamma * sa;
        ans -= alpha * &d;
    }
    Ok(ans)
}

#[test]
fn it_works() {
    let c = DVector::<f64>::from_vec(vec![-1.0, -1.0]);
    let amat = DMatrix::<f64>::from_vec(2, 2, vec![1.0, 1.0, 1.0, -1.0]);
    let b = DVector::<f64>::from_vec(vec![0.5, 1.0]);
    let x = DVector::<f64>::from_vec(vec![-2.0, -2.0]);
    let ans = karmarkar(&c, &amat, &b, &x, 0.5, 1.0e-3, 30);
    let expected = DVector::<f64>::from_vec(vec![0.23242187, 0.23242187]);
    println!("{:?}", ans);
    if let Ok(v) = ans {
        const TORELANCE: f64 = 1.0e-6;
        assert!((v[0] - expected[0]).abs() < TORELANCE);
        assert!((v[1] - expected[1]).abs() < TORELANCE);
    }
}
