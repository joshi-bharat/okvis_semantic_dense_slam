/**
 * @file dbow3_test.cpp
 * @brief Testing/generating the DBoW3 BRISK vocabulary.
 *
 * License: BSD, see https://github.com/rmsalinas/DBow3/blob/master/LICENSE.txt
 * @author Dorian Galvez-Lopez
 * @author Stefan Leutenegger
 *
 */

#include <algorithm>
#include <iostream>
#include <vector>

#include <DBoW3/DBoW3.h>

#include <boost/filesystem.hpp>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// BRISK
#include <brisk/brisk.h>

using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// \brief Load features from path: one CV_8U matrix of 48-byte BRISK descriptors per image.
/// \param path Path.
/// @param[out] features The loaded features.
void loadFeatures(const string &path, vector<cv::Mat> &features);

/// \brief Test vocabulary creation.
/// \param features Features.
void testVocCreation(const vector<cv::Mat> &features);

/// \brief Test database.
/// \param features Features.
void testDatabase(const vector<cv::Mat> &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

const size_t NIMAGES = 4; ///< number of images matched against each other in the tests

// ----------------------------------------------------------------------------

/// \brief Main
/// \param argc argc.
/// \param argv argv.
int main(int argc, char **argv)
{

  if (argc != 2) {
    std::cout << "Usage: ./" << argv[0] << " dataset-folder";
    return -1;
  }

  std::string path(argv[1]);

  vector<cv::Mat> features;
  loadFeatures(path, features);

  testVocCreation(features);

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(const string &path, vector<cv::Mat> &features)
{
  features.clear();

  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> briskDetector(36, 0, 100,700);
  brisk::BriskDescriptorExtractor briskDescriptorExtractor(false, false);

  size_t cnt = size_t(std::count_if(
          boost::filesystem::directory_iterator(path),
          boost::filesystem::directory_iterator(),
          static_cast<bool(*)(const boost::filesystem::path&)>(
                          boost::filesystem::is_regular_file)));
  cout << "Extracting BRISK features from " << cnt << " images..." << endl;
  int ctr = 0;
  for (auto it = boost::filesystem::directory_iterator(path);
      it != boost::filesystem::directory_iterator(); it++) {
    if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
      std::cout << "\r " << int(double(ctr)/double(cnt)*100.0) << "%, processing "
                << it->path().filename().string();
      cv::Mat image = cv::imread(path + "/" + it->path().filename().string(), cv::IMREAD_GRAYSCALE);
      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      briskDetector.detect(image,keypoints);
      briskDescriptorExtractor.compute(image,keypoints,descriptors);

      features.push_back(descriptors);

      ctr++;
    }
  }
  std::cout  << std::endl;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features)
{
  // branching factor and depth levels
  const int k = 9;
  const int L = 3;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;

  DBoW3::Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  DBoW3::BowVector v1, v2;
  const size_t n = std::min(NIMAGES, features.size());
  for(size_t i = 0; i < n; i++)
  {
    voc.transform(features[i], v1);
    for(size_t j = 0; j < n; j++)
    {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk, as text (yml) so it stays readable and DBoW2-compatible
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz", false);
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<cv::Mat> &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  DBoW3::Vocabulary voc("small_voc.yml.gz");

  DBoW3::Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  const size_t n = std::min(NIMAGES, features.size());
  for(size_t i = 0; i < n; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  DBoW3::QueryResults ret;
  for(size_t i = 0; i < n; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  DBoW3::Database db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}
