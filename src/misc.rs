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

pub(crate) struct ConfusionMatrix {
    /// true-positives
    pub(crate) tp: i32,

    /// false-negatives
    pub(crate) fne: i32,

    /// true-negatives
    pub(crate) tn: i32,

    /// false-positives
    pub(crate) fp: i32,
}

impl ConfusionMatrix {
    pub fn new<T: std::cmp::PartialOrd + Clone>(
        threshold: T,
        same: &Vec<T>,
        diff: &Vec<T>,
    ) -> Self {
        let tp = same.iter().filter(|x| **x <= threshold).count() as i32;
        let fne = same.len() as i32 - tp;
        let tn = diff.iter().filter(|x| **x > threshold).count() as i32;
        let fp = diff.len() as i32 - tn;
        Self { tp, fne, tn, fp }
    }

    pub fn amount_false(&self) -> i32 {
        self.fne + self.fp
    }

    pub fn amount_pos(&self) -> i32 {
        self.tp + self.tn
    }

    pub fn false_discovery_rate(&self) -> f32 {
        self.fp as f32 / (self.fp + self.tp) as f32
    }

    pub fn false_omission_rate(&self) -> f32 {
        self.fne as f32 / (self.fne + self.tn) as f32
    }
}
