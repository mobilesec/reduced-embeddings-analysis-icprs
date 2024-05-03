// Copyright (C) 2024  Johannes Kepler University Linz, Institute of Networks and Security
// Copyright (C) 2024  CDL Digidow <https://www.digidow.eu/>
//
// Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
// the European Commission - subsequent versions of the EUPL (the "Licence").
// You may not use this work except in compliance with the Licence.
//
// You should have received a copy of the European Union Public License along
// with this program.  If not, you may obtain a copy of the Licence at:
// <https://joinup.ec.europa.eu/software/page/eupl>
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Licence is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the Licence for the specific language governing permissions and
// limitations under the Licence.

mod arcface;
mod cplfw;
mod lfw;
mod misc;

use crate::cplfw::Cplfw;
use crate::misc::ConfusionMatrix;
use crate::{arcface::Recognition, lfw::Lfw};
use itertools::Itertools;
use kdam::tqdm;
use pico_args::Arguments;
use rand::seq::SliceRandom;
use std::{fs::File, io::Write};

type IsSamePerson = bool;

/// Defines the dataset, which can be used for both LFW and CPLFW dataset. It automatically caches
/// calculated embeddings.
pub trait Dataset {
    fn embeddings(
        &self,
        rec: &mut crate::arcface::Recognition,
    ) -> Vec<(IsSamePerson, Vec<f32>, Vec<f32>)>;
    fn images(&self) -> Vec<String>;
    fn name(&self) -> String;

    fn cache(&self, rec: &mut crate::arcface::Recognition) {
        for filename in tqdm!(self.images().iter()) {
            rec.cache_img(&filename.into());
        }
    }
}

/// Computes confusion matrix for a given set of distances.
struct Result<T>
where
    T: PartialOrd + Clone,
{
    same: Vec<T>,
    diff: Vec<T>,
}

impl<T: PartialOrd + Clone + std::fmt::Display + Copy + std::fmt::Debug> Result<T> {
    fn new() -> Self {
        Self {
            same: Vec::new(),
            diff: Vec::new(),
        }
    }

    /// Adds the calculated distance of two images of the same person.
    fn add_same(&mut self, dist: T) {
        self.same.push(dist);
    }

    /// Adds the calculated distance of two images of different people.
    fn add_diff(&mut self, dist: T) {
        self.diff.push(dist);
    }

    fn amount_false(&self, threshold: T) -> i32 {
        ConfusionMatrix::new(threshold, &self.same, &self.diff).amount_false()
    }

    fn calc(&self) -> String {
        let (best_threshold, confusion_matrix) = self.get_confusion_matrix();

        let best_amount_fp = confusion_matrix.fp;
        let best_amount_fn = confusion_matrix.fne;

        format!("{best_threshold};{best_amount_fp};{best_amount_fn}")
    }

    fn get_confusion_matrix(&self) -> (T, ConfusionMatrix) {
        //All possible thresholds
        let mut thresholds: Vec<T> = self
            .same
            .clone()
            .into_iter()
            .chain(self.diff.clone().into_iter())
            .collect();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        //Iterate through all thresholds, find the one that minimizes amount_false()
        let best_threshold = thresholds
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| self.amount_false(**a).cmp(&self.amount_false(**b)))
            .map(|(_, threshold)| threshold)
            .unwrap();

        (
            *best_threshold,
            ConfusionMatrix::new(*best_threshold, &self.same, &self.diff),
        )
    }

    fn calc_rel(&self) -> String {
        let (best_threshold, conf) = self.get_confusion_matrix();

        format!(
            "{best_threshold};{};{}",
            conf.false_discovery_rate(),
            conf.false_omission_rate()
        )
    }

    fn calc_return_false(&self) -> i32 {
        //All possible thresholds
        let mut thresholds: Vec<T> = self
            .same
            .clone()
            .into_iter()
            .chain(self.diff.clone().into_iter())
            .collect();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        //Iterate through all thresholds, find the one that minimizes amount_false()
        let best_threshold = thresholds
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| self.amount_false(**a).cmp(&self.amount_false(**b)))
            .map(|(_, threshold)| threshold)
            .unwrap();

        let confusion_matrix = ConfusionMatrix::new(*best_threshold, &self.same, &self.diff);
        let best_amount_fp = confusion_matrix.fp;
        let best_amount_fn = confusion_matrix.fne;

        best_amount_fn + best_amount_fp
    }
}

fn truncate_embeddings(data: Box<dyn Dataset>, rec: &mut Recognition) {
    println!("embedding_dimensions;optimal_threshold_used;fp;fn");
    for i in (1..513).rev() {
        let mut result = Result::new();
        for (same_person, mut emb1, mut emb2) in data.embeddings(rec) {
            emb1.truncate(i);
            emb2.truncate(i);

            let dist = emb1
                .iter()
                .zip(emb2.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();

            if same_person {
                result.add_same(dist);
            } else {
                result.add_diff(dist);
            }
        }

        println!("{i};{}", result.calc());
    }
}

fn truncate_embeddings_rel(data: Box<dyn Dataset>, rec: &mut Recognition) {
    println!("embedding_dimensions;optimal_threshold_used;fp;fn");
    for i in (1..513).rev() {
        let mut result = Result::new();
        for (same_person, mut emb1, mut emb2) in data.embeddings(rec) {
            emb1.truncate(i);
            emb2.truncate(i);

            let dist = emb1
                .iter()
                .zip(emb2.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>();

            if same_person {
                result.add_same(dist);
            } else {
                result.add_diff(dist);
            }
        }

        println!("{i};{}", result.calc_rel());
    }
}

fn random_dims(data: Box<dyn Dataset>, rec: &mut Recognition, amount_dimensions: usize) {
    println!("amount_dimensions;indices;optimal_threshold_used;fp;fn");
    for _ in 1..101 {
        let mut rng = rand::thread_rng();

        let mut indices = (0..512).collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        indices = indices[..amount_dimensions].to_vec();

        let mut result = Result::new();
        for (same_person, emb1, emb2) in data.embeddings(rec) {
            let mut dist = 0.;
            for &index in &indices {
                dist += (emb1[index] - emb2[index]) * (emb1[index] - emb2[index]);
            }
            if same_person {
                result.add_same(dist);
            } else {
                result.add_diff(dist);
            }
        }
        println!("{amount_dimensions};{:?};{}", indices, result.calc());
    }
}

fn random_dims_full(data: Box<dyn Dataset>, rec: &mut Recognition) {
    println!("amount_dimensions;indices;optimal_threshold_used;fp;fn");
    for amount_dimensions in (1..513).rev() {
        let mut rng = rand::thread_rng();

        let mut indices = (0..512).collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        indices = indices[..amount_dimensions].to_vec();

        let mut result = Result::new();
        for (same_person, emb1, emb2) in data.embeddings(rec) {
            let mut dist = 0.;
            for &index in &indices {
                dist += (emb1[index] - emb2[index]) * (emb1[index] - emb2[index]);
            }
            if same_person {
                result.add_same(dist);
            } else {
                result.add_diff(dist);
            }
        }
        println!("{amount_dimensions};{:?};{}", indices, result.calc());
    }
}

fn best_elements_full(data: Box<dyn Dataset>, rec: &mut Recognition, amount_dim: usize) {
    let mut possible_indices = Vec::new();
    for i in 0..amount_dim {
        possible_indices.push(i);
    }

    for i in 0..possible_indices.len() {
        let mut best = (vec![&0], 999999999);
        for perm in tqdm!(possible_indices.iter().combinations(i)) {
            let mut result = Result::new();
            for (same_person, emb1, emb2) in data.embeddings(rec) {
                let mut dist = 0.;
                for index in &perm {
                    dist += (emb1[**index] - emb2[**index]) * (emb1[**index] - emb2[**index]);
                }
                if same_person {
                    result.add_same(dist);
                } else {
                    result.add_diff(dist);
                }
            }
            if result.calc_return_false() < best.1 {
                best = (perm.clone(), result.calc_return_false());
            }
        }
        println!(
            "Best perm with {i} elements: {:?} with a total amount of errors of {}",
            best.0, best.1
        );
    }
}

fn quant(data: Box<dyn Dataset>, rec: &mut Recognition) {
    //Quantitize to integer
    let mut result = Result::new();

    let mut counter = 0;
    for (same_person, mut emb1, mut emb2) in data.embeddings(rec) {
        let dist = emb1
            .iter()
            .zip(emb2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>();

        if same_person {
            result.add_same(dist);
        } else {
            result.add_diff(dist);
        }
    }
    println!("Original f32->{}", result.calc());

    println!("scale;min-value;max-value;threshold;fp;fn");

    for i in 1..200 {
        let scale = i as f32;
        let mut min_value = std::i32::MAX;
        let mut max_value = std::i32::MIN;
        let mut result = Result::new();
        for (same_person, mut emb1, mut emb2) in data.embeddings(rec) {
            let emb1: Vec<i32> = emb1.iter().map(|&x| (x * scale) as i32).collect();
            min_value = std::cmp::min(min_value, emb1.iter().min().cloned().unwrap());
            max_value = std::cmp::max(max_value, emb1.iter().max().cloned().unwrap());

            let emb2: Vec<i32> = emb2.iter().map(|&x| (x * scale) as i32).collect();
            min_value = std::cmp::min(min_value, emb2.iter().min().cloned().unwrap());
            max_value = std::cmp::max(max_value, emb2.iter().max().cloned().unwrap());

            let dist = emb1
                .iter()
                .zip(emb2.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<i32>();

            if same_person {
                result.add_same(dist);
            } else {
                result.add_diff(dist);
            }
        }
        println!("{scale};{min_value};{max_value};{}", result.calc());
    }
}

fn proposed(data: Box<dyn Dataset>, rec: &mut Recognition) {
    let mut result = Result::new();
    for (same_person, mut emb1, mut emb2) in data.embeddings(rec) {
        let emb1: Vec<i8> = emb1.iter().map(|&x| (x * 70.) as i8).collect();
        let emb2: Vec<i8> = emb2.iter().map(|&x| (x * 70.) as i8).collect();

        let mut dist = 0;
        for index in [
            7, 9, 11, 21, 23, 30, 33, 35, 60, 61, 68, 84, 87, 92, 100, 120, 133, 134, 136, 156,
            163, 165, 167, 172, 180, 193, 202, 208, 209, 210, 211, 220, 241, 249, 262, 264, 265,
            268, 276, 279, 280, 281, 283, 294, 308, 322, 324, 325, 327, 338, 354, 360, 364, 366,
            371, 382, 408, 420, 421, 427, 433, 458, 464, 469, 470, 478, 479, 485, 488, 490,
        ] {
            dist += (emb1[index] - emb2[index]) as i32 * (emb1[index] - emb2[index]) as i32;
        }

        if same_person {
            result.add_same(dist);
        } else {
            result.add_diff(dist);
        }
    }
    println!("{}", result.calc());
}

fn best_elements_greedy(data: Box<dyn Dataset>, rec: &mut Recognition, amount_dim: usize) {
    let mut fixed: Vec<usize> = Vec::new();

    for i in 1..amount_dim + 1 {
        let mut best = (0, 999999999);
        let mut to_potentially_add: Vec<usize> = (0..amount_dim).map(|x| x as usize).collect();
        to_potentially_add.retain(|&x| !fixed.contains(&x));

        for to_add in to_potentially_add {
            let mut perm: Vec<usize> = fixed.clone();
            perm.push(to_add);
            let mut result = Result::new();
            for (same_person, emb1, emb2) in data.embeddings(rec) {
                let mut dist = 0.;
                for index in &perm {
                    dist += (emb1[*index] - emb2[*index]) * (emb1[*index] - emb2[*index]);
                }
                if same_person {
                    result.add_same(dist);
                } else {
                    result.add_diff(dist);
                }
            }
            if result.calc_return_false() < best.1 {
                best = (to_add, result.calc_return_false());
            }
            if i == 1 {
                println!(
                    "Perm with {i} elements: {to_add} with a total amount of errors of {}",
                    result.calc_return_false()
                );
            }
        }
        fixed.push(best.0);
        println!(
            "Best perm with {i} elements: {:?} with a total amount of errors of {}",
            fixed, best.1
        );
    }
}

fn heatmap(data: Box<dyn Dataset>, rec: &mut Recognition, amount_dim: usize) {
    println!("idx;neg_impact");
    let mut impact_index = vec![0_f32; amount_dim];
    for (same_person, emb1, emb2) in data.embeddings(rec) {
        for index in 0..amount_dim {
            let cur_impact = (emb1[index] - emb2[index]) * (emb1[index] - emb2[index]);
            match same_person {
                true => impact_index[index] -= cur_impact,
                false => impact_index[index] += cur_impact,
            }
        }
    }

    //Normalize to [0;1] range
    let min = *impact_index
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();
    let max = *impact_index
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap();

    // Normalize
    for i in 0..impact_index.len() {
        impact_index[i] = (impact_index[i] - min) / (max - min);
    }

    for (idx, value) in impact_index.iter().enumerate() {
        println!("{};{}", idx, value);
    }
}

fn extract_emb(data: Box<dyn Dataset>, rec: &mut Recognition) {
    let mut full = Vec::new();
    let mut comp = Vec::new();
    let indices = [
        7, 9, 11, 21, 23, 30, 33, 35, 60, 61, 68, 84, 87, 92, 100, 120, 133, 134, 136, 156, 163,
        165, 167, 172, 180, 193, 202, 208, 209, 210, 211, 220, 241, 249, 262, 264, 265, 268, 276,
        279, 280, 281, 283, 294, 308, 322, 324, 325, 327, 338, 354, 360, 364, 366, 371, 382, 408,
        420, 421, 427, 433, 458, 464, 469, 470, 478, 479, 485, 488, 490,
    ];
    for (same_person, emb1, emb2) in data.embeddings(rec) {
        full.push(serde_json::to_string(&format!("{};{};{}", same_person, emb1, emb2)).unwrap());

        let emb1: Vec<i8> = emb1.iter().map(|&x| (x * 70.) as i8).collect();
        let emb2: Vec<i8> = emb2.iter().map(|&x| (x * 70.) as i8).collect();

        let emb1: Vec<i8> = indices.iter().map(|&i| emb1[i]).collect();
        let emb2: Vec<i8> = indices.iter().map(|&i| emb2[i]).collect();
        comp.push(serde_json::to_string(&(same_person, emb1, emb2)).unwrap());
    }

    let mut file = File::create("embeddings_full.json").unwrap();
    file.write_all(full).unwrap();

    let mut file = File::create("embeddings_70.json").unwrap();
    file.write_all(comp).unwrap();
}

fn expect_amount(args: &mut Arguments) -> usize {
    if let Ok(amount) = args.value_from_str::<&str, usize>("--amount") {
        return amount;
    } else {
        panic!("Expected a number how many dimensions should be used: --amount <number>");
    }
}

fn main() {
    let mut args = pico_args::Arguments::from_env();

    let data: Box<dyn Dataset> = match args.value_from_str::<&str, String>("--data") {
        Ok(d) if d == "easy" => {
            if let Ok(path) = args.value_from_str::<&str, String>("--lfwpath") {
                Box::new(Lfw::new("data/lfw-pairs.txt", path).unwrap())
            } else {
                panic!("Expected --lfwpath argument");
            }
        }
        Ok(d) if d == "hard" => {
            if let Ok(path) = args.value_from_str::<&str, String>("--cplfwpath") {
                Box::new(Cplfw::new("data/pairs_CPLFW.txt", path).unwrap())
            } else {
                panic!("Expected --cplfwpath argument");
            }
        }
        _ => {
            panic!("Expected --data argument, possible values: easy, hard");
        }
    };

    let mut rec = Recognition::default(&data.name());

    if let Ok(action) = args.opt_value_from_str::<&str, String>("--action") {
        match action {
            Some(a) if a == "cache" => data.cache(&mut rec),
            Some(a) if a == "extract-emb" => extract_emb(data, &mut rec),
            Some(a) if a == "truncate-embedding-size" => truncate_embeddings(data, &mut rec),
            Some(a) if a == "truncate-embedding-size-rel" => {
                truncate_embeddings_rel(data, &mut rec)
            }
            Some(a) if a == "random-dimensions" => {
                random_dims(data, &mut rec, expect_amount(&mut args))
            }
            Some(a) if a == "random-dimensions-full" => random_dims_full(data, &mut rec),
            Some(a) if a == "best-elements-full" => {
                best_elements_full(data, &mut rec, expect_amount(&mut args))
            }
            Some(a) if a == "best-elements-greedy" => {
                best_elements_greedy(data, &mut rec, expect_amount(&mut args))
            }
            Some(a) if a == "heatmap" => heatmap(data, &mut rec, expect_amount(&mut args)),
            Some(a) if a == "quant" => quant(data, &mut rec),
            Some(a) if a == "proposed" => proposed(data, &mut rec),
            _ => {
                panic!("Expected --action argument must have one of these values: cache, truncate-embedding-size, random-dimensions, random-dimensions-full, best-elements-full, best-elements-greedy, heatmap, quant, extract-full-emb");
            }
        }
    }
}
