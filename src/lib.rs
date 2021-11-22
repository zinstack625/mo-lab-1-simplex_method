use log::debug;
use ndarray::{Array1, Array2, ArrayView1};
use std::fmt::Display;

#[derive(Debug)]
pub enum SimplexError {
    UnableToCalculateError,
    UnlimitedError,
    NoSolutionsError,
    InvalidDataError,
}

pub struct Table {
    pub table: Array2<f64>,
    pub base_var: Vec<String>,
    pub supp_var: Vec<String>,
}

impl Display for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in self.supp_var.iter() {
            write!(f, "\t| {}\t", i)?;
        }
        write!(f, "\n")?;
        for i in 0..self.table.nrows() {
            write!(f, "{}", self.base_var[i])?;
            for j in self.table.row(i).iter() {
                write!(f, "\t| {:.7}", j)?;
            }
            write!(f, "\n")?;
        }
        writeln!(f, "")?; // empty line
        self.print_function(f)?;
        Ok(())
    }
}

impl Table {
    pub fn new(
        mut constr_coeff: Array2<f64>,
        constr_val: Array1<f64>,
        mut func_coeff: Array1<f64>,
        minimisation_task: bool,
    ) -> Table {
        if minimisation_task {
            for i in func_coeff.iter_mut() {
                *i *= -1f64;
            }
        }
        constr_coeff.push_row(ArrayView1::from(&func_coeff));
        let mut constr_val_vec = constr_val.to_vec();
        while constr_val_vec.len() < constr_coeff.nrows() {
            constr_val_vec.push(0f64);
        }
        let constr_val = Array1::from_vec(constr_val_vec);
        constr_coeff
            .push_column(ArrayView1::from(&constr_val))
            .unwrap();
        let mut supp_var = Vec::<String>::with_capacity(constr_coeff.ncols() - 1);
        for i in 1..constr_coeff.ncols() {
            supp_var.push(i.to_string());
        }
        supp_var.push("S".to_string());
        let mut base_var = Vec::<String>::with_capacity(constr_coeff.nrows());
        for i in 0..constr_coeff.nrows() - 1 {
            base_var.push((i + constr_coeff.ncols()).to_string());
        }
        base_var.push("F".to_string());
        Table {
            table: constr_coeff,
            base_var,
            supp_var,
        }
    }
    pub fn optimise(&mut self) -> Result<(), crate::SimplexError> {
        debug!("Beginning table:\n{}", self);
        loop {
            let negative_row = self.find_in_free_column();
            if negative_row.is_none() {
                break;
            }
            let negative_row = negative_row.unwrap();
            debug!("Found negative free coeff in row {}", negative_row);
            self.make_acceptable(negative_row)?;
            debug!("Making acceptable:\n{}", self);
        }
        // begin transformation
        while !self.check_optimised() {
            self.iterate()?;
            debug!("Iteration:\n{}\n", self);
        }
        Ok(())
    }

    fn check_optimised(&self) -> bool {
        for i in self.table.row(self.table.nrows() - 1).iter().enumerate() {
            // don't check for the last column (with free coeffs)
            if i.0 >= self.table.ncols() - 1 {
                break;
            }
            if i.1.is_sign_positive() {
                return false;
            }
        }
        return true;
    }

    fn find_in_free_column(&self) -> Option<usize> {
        for i in self.table.column(self.table.ncols() - 1).iter().enumerate() {
            // don't check for the last column (with free coeffs)
            if i.0 >= self.table.nrows() - 1 {
                break;
            }
            if i.1.is_sign_negative() {
                return Some(i.0);
            }
        }
        return None;
    }

    fn make_acceptable(&mut self, negative: usize) -> Result<(), SimplexError> {
        let j = {
            let mut index = None;
            for i in self.table.row(negative).iter().enumerate() {
                // don't check for the last column (with free coeffs)
                if i.0 >= self.table.ncols() - 1 {
                    break;
                }
                if i.1.is_sign_negative() {
                    index = Some(i.0);
                    break;
                }
            }
            if index.is_none() {
                return Err(SimplexError::NoSolutionsError);
            }
            index.unwrap()
        };
        if let Some(i) = self.find_pivot_row(j) {
            debug!("Transforming on pivot i: {}\tj: {}", i, j);
            self.transform((i, j));
            std::mem::swap(&mut self.base_var[j], &mut self.supp_var[i]);
            return Ok(());
        } else {
            return Err(SimplexError::UnableToCalculateError);
        }
    }

    fn iterate(&mut self) -> Result<(), SimplexError> {
        let (i, j) = self.find_pivot()?;
        debug!("Current pivot: i: {}\tj:{}", i, j);
        self.transform((i, j));
        std::mem::swap(&mut self.base_var[j], &mut self.supp_var[i]);
        Ok(())
    }

    fn find_pivot(&self) -> Result<(usize, usize), SimplexError> {
        let j = {
            let mut index = None;
            for j in self.table.row(self.table.nrows() - 1).iter().enumerate() {
                if j.1.is_sign_positive() {
                    index = Some(j.0);
                    break;
                }
            }
            if index.is_none() {
                return Err(SimplexError::UnableToCalculateError);
            }
            index.unwrap()
        };
        let mut has_positive = false;
        for i in self.table.column(j) {
            if i.is_sign_positive() {
                has_positive = true;
                break;
            }
        }
        if !has_positive {
            return Err(SimplexError::UnlimitedError);
        }
        if let Some(i) = self.find_pivot_row(j) {
            return Ok((i, j));
        } else {
            return Err(SimplexError::UnlimitedError);
        }
    }

    fn find_pivot_row(&self, column: usize) -> Option<usize> {
        let mut min = None;
        let mut min_index = None;
        for i in self.table.column(self.table.ncols() - 1).iter().enumerate() {
            // don't check the function coeffs
            if i.0 >= self.table.nrows() - 1 {
                break;
            }
            let relation = i.1 / self.table[[i.0, column]];
            if (min.is_none() || relation < min.unwrap()) && relation.is_sign_positive() {
                min = Some(relation);
                min_index = Some(i.0);
            }
        }
        min_index
    }

    fn transform(&mut self, pivot: (usize, usize)) {
        let pivot_cpy = self.table[[pivot.0, pivot.1]];
        for i in 0..self.table.nrows() {
            for j in 0..self.table.ncols() {
                if i == pivot.0 || j == pivot.1 {
                    continue;
                }
                self.table[[i, j]] -=
                    self.table[[pivot.0, j]] * self.table[[i, pivot.1]] / pivot_cpy;
            }
        }
        for i in self.table.row_mut(pivot.0) {
            *i /= pivot_cpy;
        }
        for i in self.table.column_mut(pivot.1) {
            *i /= -pivot_cpy;
        }
        self.table[[pivot.0, pivot.1]] = 1f64 / pivot_cpy;
    }
    pub fn print_function(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in self.table.column(self.table.ncols() - 1).iter().enumerate() {
            if i.0 >= self.table.nrows() - 1 {
                break;
            }
            writeln!(f, "X_{} = {}", self.base_var[i.0], i.1)?;
        }
        write!(f, "F = {}", self.table.column(self.table.ncols() - 1).last().unwrap())?;
        Ok(())
    }
}
