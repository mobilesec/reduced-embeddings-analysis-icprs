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

use std::{collections::HashMap, path::PathBuf};

use face::{
    detection::{retinaface, DetectionError},
    recognition::arcface,
};

type Embedding = Vec<f32>;

#[derive(Debug)]
/// Defines all possible error for recognition.
pub enum Error {
    /// Data could not be deserialized
    DeserializeError(serde_json::Error),

    /// Retinaface threw an error
    RetinafaceError(retinaface::RetinafaceError),

    ///Fastdet threw an error
    FastdetError(DetectionError),
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::DeserializeError(value)
    }
}

impl From<retinaface::RetinafaceError> for Error {
    fn from(value: retinaface::RetinafaceError) -> Self {
        Self::RetinafaceError(value)
    }
}
impl From<DetectionError> for Error {
    fn from(value: DetectionError) -> Self {
        Self::FastdetError(value)
    }
}

/// Caches arcfaces' results
pub struct Recognition {
    /// Calculated embeddings for `filepath`s
    pub emb: HashMap<PathBuf, Embedding>,

    /// Face detection model
    pub face: retinaface::Retinaface,

    /// Face recognition model
    pub arcface: arcface::ArcFace,

    /// Path to cache file. If None, nothing is cached
    cache_path: Option<PathBuf>,

    #[cfg(test)]
    amount_new_calculated: u32,
}

impl Recognition {
    //TODO: impl Default-trait
    pub fn default(name: &str) -> Self {
        Recognition::new(
            Some(PathBuf::from(format!("data/cache-{}-250x250.json", name))),
            "data/models/retinaface-250x250.tflite",
            "data/models/retinaface-anchors-250x250.json",
            "data/models/arcface.tflite",
        )
        .unwrap()
    }

    /// Creates a new `Recognition` struct.
    ///
    /// # Arguments
    /// - `path` if supplied, all calculations are cached there. If non existing file is supplied,
    /// the file will be created.
    /// - `retinaface_model`: TF Lite model file of retinaface
    /// - `retinaface_anchor`: Anchor settings used for `retinaface_model`
    /// - `arcface_model`: TF Lite model file of arcface
    ///
    /// # Errors
    /// - `Error::DeserializeError` returned, if supplied cache file is not deserializable
    /// - `Error::RetinafaceError` returned, if creation of Retinaface threw an error, e.g. because
    /// models could not be found
    pub fn new(
        path: Option<PathBuf>,
        retinaface_model: &str,
        retinaface_anchors: &str,
        arcface_model: &str,
    ) -> Result<Self, Error> {
        let mut emb = HashMap::new();
        if let Some(path) = &path {
            emb = match std::fs::read_to_string(path) {
                Ok(data) => serde_json::from_str(&data)?,
                Err(_) => HashMap::new(),
            };
        }

        log::info!("Loaded {} embeddings from cache", emb.len());

        //let face = face::fast::FastInference::new(0.7).unwrap();
        let face =
            face::detection::retinaface::Retinaface::new(retinaface_model, retinaface_anchors)?;
        let arcface = face::recognition::arcface::ArcFace::new(arcface_model);

        Ok(Self {
            emb,
            face,
            arcface,
            cache_path: path,
            #[cfg(test)]
            amount_new_calculated: 0,
        })
    }

    fn add(&mut self, filename: PathBuf, emb: Embedding) {
        self.emb.insert(filename, emb);
        if let Some(path) = &self.cache_path {
            let j = serde_json::to_string(&self.emb).unwrap();
            std::fs::write(path, j).expect("Unable to write file");
        }
    }

    /// Retrieves an cached embedding
    pub fn get(&self, filename: PathBuf) -> Option<&Vec<f32>> {
        self.emb.get(&filename)
    }

    /// Caches an embedding.
    ///
    /// If `filename` is already in the cache, nothing is done.
    /// If multiple faces are detected, the one most in the middle is used.
    ///
    /// # Panics
    /// - Panics if `filename` can't be read (panic occurs inside `img_read` macro
    pub fn cache_img(&mut self, filename: &PathBuf) {
        if !self.emb.contains_key(filename) {
            //#[cfg(test)] uncommenting these two lines gives compiler error ?!? TODO GS
            //self.amount_new_calculated += 1;
            let img = face::img_read!(&filename);
            let res = self.face.inference(&img).unwrap();
            if !res.is_empty() {
                let idx_most_centered_face = res
                    .iter()
                    .map(|a| (125. - a.landmarks.nose.x).abs() + (125. - a.landmarks.nose.y).abs())
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap();

                let image = face::warp::Warp::perform(&img, &res[idx_most_centered_face].landmarks);

                let emb = self.arcface.calc_emb(&image).unwrap(); // unwrap okay; because it only fails if network is wrong
                self.add(filename.clone(), emb.into());
            } else {
                println!("Ignored {filename:?}, {} faces found", res.len());
            }
        }
    }
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    #[test]
//    fn test_no_cachefile() {
//        let mut r = Recognition::new(
//            None,
//            "data/models/retinaface-250x250.tflite",
//            "data/models/retinaface-anchors-250x250.json",
//            "data/models/arcface.tflite",
//        )
//        .unwrap();
//
//        r.cache_img(&PathBuf::from(
//            "data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
//        ));
//        r.cache_img(&PathBuf::from(
//            "data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg",
//        ));
//
//        assert_eq!(1, r.emb.len());
//
//        //assert_eq!(1, r.amount_new_calculated);
//    }
//}
