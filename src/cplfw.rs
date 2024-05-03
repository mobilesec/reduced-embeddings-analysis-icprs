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

use crate::Dataset;

/// Represents an image pair of the CPLFW dataset
pub struct Cplfw {
    pair: Vec<(bool, String, String)>,
}

impl Cplfw {
    pub fn new(pairs_file: &str, basepath: String) -> Result<Self, ()> {
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .delimiter(b'\t')
            .from_path(pairs_file)
            .unwrap();

        let mut pairs = Vec::new();

        let mut prev: Option<(bool, String)> = None;
        for record in reader.records() {
            let name: String = record.unwrap()[0].into();
            let splitted = name.split(" ").collect::<Vec<_>>();

            match prev {
                Some((same, ref name)) => {
                    pairs.push((
                        same,
                        format!("{basepath}/{}", name.clone()),
                        format!("{basepath}/{}", splitted[0]).into(),
                    ));

                    prev = None;
                }
                None => {
                    prev = Some((splitted[1] == "1", splitted[0].into()));
                }
            }
        }

        Ok(Self { pair: pairs })
    }
}

impl Dataset for Cplfw {
    fn embeddings(&self, rec: &mut crate::arcface::Recognition) -> Vec<(bool, Vec<f32>, Vec<f32>)> {
        self.pair
            .iter()
            .filter_map(|(same_person, path1, path2)| {
                if let Some(emb1) = rec.get(path1.into()) {
                    if let Some(emb2) = rec.get(path2.into()) {
                        Some((*same_person, emb1.clone(), emb2.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    fn images(&self) -> Vec<String> {
        let mut ret = Vec::new();
        for (_, path1, path2) in &self.pair {
            ret.push(path1.clone());
            ret.push(path2.clone());
        }
        ret
    }

    fn name(&self) -> String {
        "cplfw".into()
    }
}
