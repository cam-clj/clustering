(ns clustering.k-means
  (:require [quil.core :as q]
            [clojure.java.io :as io]
            [incanter.distributions :as dist]
            [incanter.charts :refer [scatter-plot]]
            [incanter.core :refer [view]]))

(defn write-dataset
  [ds path]
  (with-open [writer (io/writer path)]
    (binding [*out* writer]
      (doseq [[x y] ds]
        (println x y)))))

(defn get-ranges
  "Given a seq of n-dimensional vectors, `xs`, find the minimum and
  maximum values in each dimension. Return a sequence of vectors of max/min."
  [xs]
  (when (seq xs)
    (for [i (range (count (first xs)))]
      [(reduce min (map #(nth % i) xs))
       (reduce max (map #(nth % i) xs))])))

(defn generate-random-point
  "Given a vector of n min/max ranges, generate a random point in
  n-dimensional space, bounded by min/max in each dimension."
  [ranges]
  (mapv (fn [[lo hi]] (+ lo (rand (- hi lo)))) ranges))

(defn random-centroids
  "Given a set of points `xs`, return `k` random points in
  n-dimensional space, bounded by the max/min of each dimension in
  `xs`."
  [k xs]
  (when-let [ranges (get-ranges xs)]
    (repeatedly k #(generate-random-point ranges))))

(defn square
  [x]
  (* x x))

(defn distance
  [p1 p2]
  (Math/sqrt (reduce + (map (comp square -) p1 p2))))

(defn nearest-centroid
  [p centroids]
  (apply min-key #(distance p %) centroids))

(defn mean
  [xs]
  (let [c (count xs)]
    (if (> c 0)
      (/ (reduce + xs) c)
      0)))

(defn centroid
  [xs]
  (when (seq xs)
    (mapv mean (map (fn [i] (map #(nth % i) xs)) (range (count (first xs)))))))

(defn make-cluster
  [xs centroids]
  (let [xs (vec xs)
        cluster (group-by #(nearest-centroid (xs %) centroids) (range (count xs)))]
    (zipmap centroids (map (fn [centroid] (set (get cluster centroid))) centroids))))

(defn next-cluster
  [xs cluster]
  (let [xs (vec xs)
        centroids (map #(if (seq %) (centroid (map xs %)) (first (random-centroids 1 xs))) (vals cluster))]
    (make-cluster xs centroids)))

(defn stable?
  [[c1 c2]]
  (= (set (vals c1)) (set (vals c2))))

(defn lazy-seq-cluster
  [k xs]
  (iterate (partial next-cluster xs) (make-cluster xs (random-centroids k xs))))

(defn k-means-cluster
  [k xs]
  (let [cluster (ffirst (filter stable? (partition 2 1 (lazy-seq-cluster k xs))))]
    cluster))

;; Testing...

(defn generate-test-data
  "Generate `num-points` random points in `num-dimensions`."
  ([]
     (generate-test-data 2))
  ([num-dimensions]
     (generate-test-data num-dimensions 100))
  ([num-dimensions num-points]
     (let [ranges (repeat num-dimensions [0 1])]
       (repeatedly num-points #(generate-random-point ranges)))))

(defn generate-clustered-test-data
  [& {:keys [x-max x-sd y-max y-sd num-clusters num-points-per-cluster]
      :or {x-max 1000
           y-max 1000
           x-sd  20
           y-sd  20
           num-clusters 3
           num-points-per-cluster 100}}]
  (letfn [(int-in-range [x m]
            (max 0 (min m (int x))))
          (generate-clustered-points
            [[x y]]
            (let [x-dist (dist/normal-distribution x x-sd)
                  y-dist (dist/normal-distribution y y-sd)]
              (repeatedly num-points-per-cluster #(vector (int-in-range (dist/draw x-dist) x-max)
                                                          (int-in-range (dist/draw y-dist) y-max)))))]
    (mapcat generate-clustered-points (repeatedly num-clusters #(vector (rand-int x-max) (rand-int y-max))))))

;;(view (scatter-plot (map first points) (map second points))

;; Quil stuff

(def centroid-side-len 20)
(def point-radius 5)

;; Red, Green, Blue, Cyan
(def colours
  [[255 0 0]
   [0 0 255]
   [0 255 0]
   [0 255 255]])

(def background-colour 200)

(defn draw-centroid
  [[x y]]
  (q/triangle (- x (* 0.5 centroid-side-len))
              (- y (* 0.5 centroid-side-len (Math/tan (/ Math/PI 6))))
              x
              (+ y (/ (* 0.5 centroid-side-len) (Math/cos (/ Math/PI 6))))
              (+ x (* 0.5 centroid-side-len))
              (- y (* 0.5 centroid-side-len (Math/tan (/ Math/PI 6))))))

(defn draw-points
  [points]
  (doseq [[x y] points]
    (q/ellipse x y point-radius point-radius)))

(defn draw-cluster
  [points clusters prev]
  (when prev
    (q/fill background-colour)
    (q/stroke background-colour)
    (doseq [centroid prev] (draw-centroid centroid)))  
  (doall (map (fn [centroid colour]
                (apply q/fill colour)
                (apply q/stroke colour)
                (draw-centroid centroid)
                (draw-points (map (vec points) (clusters centroid))))
              (sort (keys clusters))
              colours)))

(defn animate-k-means-clustering
  [num-clusters]
  (let [points (generate-clustered-test-data :num-clusters num-clusters)
        state  (atom {:steps (lazy-seq-cluster num-clusters points)})]
    (q/sketch
     :title "K-means clustering"
     :setup (fn [] (q/smooth) (q/frame-rate 1) (q/background background-colour))
     :draw  (fn []
              (let [this-step (first (@state :steps))]
                (draw-cluster points this-step (@state :prev))
                (swap! state #(-> %
                               (assoc :prev (keys this-step))
                               (update-in [:steps] rest))))
              (Thread/sleep 1000))
     :size  [1000 1000])))
