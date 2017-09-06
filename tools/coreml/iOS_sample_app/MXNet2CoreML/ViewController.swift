// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//
//  ViewController.swift
//  MXNet2CoreML
//
//  Created by Sebastien Menant on 8/28/17.
//

import UIKit
import Vision
import MapKit


class PredictionLocation: NSObject, MKAnnotation{
    var identifier = "Prediction location"
    var title: String?
    var coordinate: CLLocationCoordinate2D
    init(name:String,lat:CLLocationDegrees,long:CLLocationDegrees){
        title = name
        coordinate = CLLocationCoordinate2DMake(lat, long)
    }
}

class PredictionLocationList: NSObject {
    var place = [PredictionLocation]()
    override init(){
        place += [PredictionLocation(name:"1",lat: 0, long: 0)]
        place += [PredictionLocation(name:"2",lat: 1, long: 1)]
        place += [PredictionLocation(name:"3",lat: 2, long: 2)]
    }
}


class ViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var pageControl: UIPageControl!
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var mapView: MKMapView!
 
    
    var index = 0
    var pictureArray:[String] = ["1.jpg", "2.jpg", "3.jpg"]
    var pictureString:String = "nil"
    
    // Define Core ML model
    // Make sure to add the file  in the Project Navigator, and have Target Membership checked
    let model = RN1015k500()
    
    
    //MARK: - Map setup
    func resetRegion(){
        let region = MKCoordinateRegionMakeWithDistance(annotation.coordinate, 5000, 5000)
        mapView.setRegion(region, animated: true)
    }
    
    var myLatitude = ""
    var myLongitude = ""
    
    // Array of annotations
    let annotation = MKPointAnnotation()
    var places = PredictionLocationList().place
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        //Swipe configuration
        let swipeLeft = UISwipeGestureRecognizer(target: self, action: #selector(pictureSwipe))
        swipeLeft.direction = .left
        self.view.addGestureRecognizer(swipeLeft)
        
        let swipeRight = UISwipeGestureRecognizer(target: self, action: #selector(pictureSwipe))
        swipeRight.direction = .right
        self.view.addGestureRecognizer(swipeRight)
        
        pictureString = pictureArray[0]
        
        let image = UIImage(named: pictureString)
        imageView.image = image
        
        predictUsingVision(image: image!)

    }

    func predictUsingVision(image: UIImage) {
        guard let visionModel = try? VNCoreMLModel(for: model.model) else {
            fatalError("Something went wrong")
        }
        
        let request = VNCoreMLRequest(model: visionModel) { request, error in
            if let observations = request.results as? [VNClassificationObservation] {
                let top3 = observations.prefix(through: 2)
                    .map { ($0.identifier, Double($0.confidence)) }
                self.show(results: top3)
            }
        }
        
        request.imageCropAndScaleOption = .centerCrop
        
        let handler = VNImageRequestHandler(cgImage: image.cgImage!)
        try? handler.perform([request])
    }

    typealias Prediction = (String, Double)

    func show(results: [Prediction]) {
        var s: [String] = []
        for (i, pred) in results.enumerated() {
            let latLongArr = pred.0.components(separatedBy: "\t")
            myLatitude = latLongArr[1]
            myLongitude = latLongArr[2]
            s.append(String(format: "%d: %@ %@ (%3.2f%%)", i + 1, myLatitude, myLongitude, pred.1 * 100))
            places[i].title = String(i+1)
            places[i].coordinate = CLLocationCoordinate2D(latitude: CLLocationDegrees(myLatitude)!, longitude: CLLocationDegrees(myLongitude)!)
        }
        predictionLabel.text = s.joined(separator: "\n")
        
        // Map reset
        resetRegion()
        // Center on first prediction
        mapView.centerCoordinate = places[0].coordinate
        // Show annotations for the predictions on the map
        mapView.addAnnotations(places)
        // Zoom map to fit all annotations
        zoomMapFitAnnotations()
    }
    
    
    @IBAction func pictureSwipe(_ gesture: UISwipeGestureRecognizer) {
        //print("Image swipped")
        
        let updateIndex = gesture.direction == .left ? 1 : -1
        index += updateIndex
        
        if index >= pictureArray.count {
            index = 0
        } else if index < 0 {
            index = pictureArray.count - 1
        }
        let pictureString = pictureArray[index]
        self.imageView.image = UIImage(named: pictureString)
        predictUsingVision(image: self.imageView.image!)
        pageControl.currentPage = index
    }
    
    
    
    func zoomMapFitAnnotations() {
        var zoomRect = MKMapRectNull
        for annotation in mapView.annotations {
            let annotationPoint = MKMapPointForCoordinate(annotation.coordinate)
            let pointRect = MKMapRectMake(annotationPoint.x, annotationPoint.y, 0, 0)
            if (MKMapRectIsNull(zoomRect)) {
                zoomRect = pointRect
            } else {
                zoomRect = MKMapRectUnion(zoomRect, pointRect)
            }
        }
        self.mapView.setVisibleMapRect(zoomRect, edgePadding: UIEdgeInsetsMake(50, 50, 50, 50), animated: true)
    }

}

