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

use csv::StringRecord;
use serde::Deserialize;

use crate::Dataset;

#[derive(Debug)]
/// All possible Lfw Errors
pub enum Error {
    //Can't find pairs file
    CsvError(csv::Error),
}

impl From<csv::Error> for Error {
    fn from(value: csv::Error) -> Self {
        Self::CsvError(value)
    }
}

#[derive(Deserialize, Debug)]
/// Holds a pair of LFW validation images.
/// If `name2` is not set, `nr2` is from the same person as `name`
pub struct Pair {
    name: String,
    nr: String,
    name2: Option<String>,
    nr2: String,

    /// Specifies path to lfw dataset (folder containing e.g. Aaron_Eckhart)
    basepath: String,
}

impl Pair {
    fn new(s: StringRecord, basepath: String) -> Pair {
        if s.len() == 3 {
            Self {
                name: s[0].into(),
                nr: s[1].into(),
                name2: None,
                nr2: s[2].into(),
                basepath,
            }
        } else {
            Self {
                name: s[0].into(),
                nr: s[1].into(),
                name2: Some(s[2].into()),
                nr2: s[3].into(),
                basepath,
            }
        }
    }

    pub fn same_person(&self) -> bool {
        self.name2.is_none()
    }

    pub fn get_path1(&self) -> String {
        format!("{0}/{1}/{1}_{2:0>4}.jpg", self.basepath, self.name, self.nr)
    }

    pub fn get_path2(&self) -> String {
        if let Some(name2) = self.name2.clone() {
            format!("{0}/{1}/{1}_{2:0>4}.jpg", self.basepath, name2, self.nr2)
        } else {
            format!(
                "{0}/{1}/{1}_{2:0>4}.jpg",
                self.basepath, self.name, self.nr2
            )
        }
    }
}

pub struct Lfw {
    pub pairs: Vec<Pair>,
}

impl Lfw {
    pub fn new(pairs_file: &str, basepath: String) -> Result<Self, Error> {
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .delimiter(b'\t')
            .from_path(pairs_file)?;

        let mut pairs = Vec::new();

        for record in reader.records() {
            let pair = Pair::new(record.unwrap(), basepath.clone());
            pairs.push(pair);
        }

        Ok(Self { pairs })
    }
}

impl Dataset for Lfw {
    fn embeddings(&self, rec: &mut crate::arcface::Recognition) -> Vec<(bool, Vec<f32>, Vec<f32>)> {
        self.pairs
            .iter()
            .map(|p| {
                (
                    p.same_person(),
                    rec.get(p.get_path1().into()).unwrap().clone(),
                    rec.get(p.get_path2().into()).unwrap().clone(),
                )
            })
            .collect()
    }

    fn images(&self) -> Vec<String> {
        let mut ret = Vec::new();
        for pair in &self.pairs {
            ret.push(pair.get_path1());
            ret.push(pair.get_path2());
        }
        ret
    }

    fn name(&self) -> String {
        "lfw".into()
    }
}
