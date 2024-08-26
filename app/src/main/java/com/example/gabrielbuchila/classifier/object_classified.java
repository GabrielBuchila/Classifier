package com.example.gabrielbuchila.classifier;


public class object_classified{

        private  String id; // A unique identifier of the object detected by tensorflow
        private String title; //The name of the object detected by tensorflow
        private Float confidence; //the probability if the object detections are ok


        public void set_id(String new_id)
        {
            this.id=new_id;
        }

        public void set_title(String new_title)
        {
            this.title=new_title;
        }

        public void set_confidence(float new_confidence)
        {
            this.confidence=new_confidence;
        }


        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence * 100 ;
        }


    }



