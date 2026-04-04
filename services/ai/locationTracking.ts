/**
 * Seren AI — Location Diversity Tracking
 * ========================================
 * Tracks user location patterns to detect monotonous routines
 * (home → work → home only) which correlate with depression.
 *
 * Privacy: All location data stays on-device in SQLite.
 * GPS coordinates are stored for clustering but never transmitted.
 *
 * Uses expo-location for GPS and simple distance-based clustering
 * to identify unique places (home, work, novel locations).
 */

import * as Location from 'expo-location';
import { LocationVisit, LocationDiversitySummary } from './types';

// ============================================================
// Constants
// ============================================================

/** Radius in meters to consider two GPS points as the same place */
const CLUSTER_RADIUS_METERS = 100;

/** Minimum unique places to NOT be considered monotonous */
const MONOTONOUS_THRESHOLD = 2;

// ============================================================
// Haversine Distance
// ============================================================

/** Calculate distance between two GPS coordinates in meters */
export function haversineDistance(
  lat1: number, lon1: number,
  lat2: number, lon2: number,
): number {
  const R = 6371000; // Earth radius in meters
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// ============================================================
// Location Clustering
// ============================================================

interface LocationCluster {
  index: number;
  label: string;
  latitude: number;    // Centroid
  longitude: number;
  visitCount: number;
  totalTimeMs: number;
  nightTimeMs: number; // Time spent between 22:00 and 06:00
  dayTimeMs: number;   // Time spent between 08:00 and 18:00 on weekdays
}

/**
 * Cluster location visits by proximity.
 * Assigns 'home' to the cluster with most nighttime hours,
 * 'work' to the cluster with most weekday daytime hours.
 */
export function computeLocationClusters(visits: LocationVisit[]): LocationCluster[] {
  if (visits.length === 0) return [];

  const clusters: LocationCluster[] = [];

  for (const visit of visits) {
    let matched = false;
    for (const cluster of clusters) {
      const dist = haversineDistance(
        visit.latitude, visit.longitude,
        cluster.latitude, cluster.longitude,
      );
      if (dist < CLUSTER_RADIUS_METERS) {
        // Update centroid (running average)
        const total = cluster.visitCount + 1;
        cluster.latitude = (cluster.latitude * cluster.visitCount + visit.latitude) / total;
        cluster.longitude = (cluster.longitude * cluster.visitCount + visit.longitude) / total;
        cluster.visitCount = total;

        const duration = visit.departureTime - visit.timestamp;
        cluster.totalTimeMs += duration;

        // Classify time spent
        const hour = new Date(visit.timestamp).getHours();
        const day = new Date(visit.timestamp).getDay();
        if (hour >= 22 || hour < 6) {
          cluster.nightTimeMs += duration;
        }
        if (day >= 1 && day <= 5 && hour >= 8 && hour < 18) {
          cluster.dayTimeMs += duration;
        }

        matched = true;
        break;
      }
    }

    if (!matched) {
      const duration = visit.departureTime - visit.timestamp;
      const hour = new Date(visit.timestamp).getHours();
      const day = new Date(visit.timestamp).getDay();
      clusters.push({
        index: clusters.length,
        label: `place_${clusters.length}`,
        latitude: visit.latitude,
        longitude: visit.longitude,
        visitCount: 1,
        totalTimeMs: duration,
        nightTimeMs: (hour >= 22 || hour < 6) ? duration : 0,
        dayTimeMs: (day >= 1 && day <= 5 && hour >= 8 && hour < 18) ? duration : 0,
      });
    }
  }

  // Assign labels: home = most nighttime, work = most weekday daytime
  if (clusters.length > 0) {
    const homeCluster = clusters.reduce((a, b) => a.nightTimeMs > b.nightTimeMs ? a : b);
    homeCluster.label = 'home';

    const workCandidates = clusters.filter(c => c !== homeCluster);
    if (workCandidates.length > 0) {
      const workCluster = workCandidates.reduce((a, b) => a.dayTimeMs > b.dayTimeMs ? a : b);
      if (workCluster.dayTimeMs > 0) {
        workCluster.label = 'work';
      }
    }
  }

  return clusters;
}

// ============================================================
// Diversity Score Calculation
// ============================================================

/**
 * Calculate location diversity summary from visits.
 * Score formula: min(100, (uniquePlaces - 1) * 25 + novelPlaces * 15 + transitionBonus)
 */
export function calculateDiversityScore(
  visits: LocationVisit[],
  weeklyVisitHistory: LocationVisit[] = [],
): LocationDiversitySummary {
  const today = new Date().toISOString().slice(0, 10);
  const clusters = computeLocationClusters(visits);

  if (clusters.length === 0) {
    return {
      date: today,
      uniquePlacesVisited: 0,
      totalTransitions: 0,
      diversityScore: 0,
      homeTimePercent: 0,
      workTimePercent: 0,
      novelPlaces: 0,
      isMonotonous: true,
    };
  }

  const uniquePlaces = clusters.length;
  const totalTime = clusters.reduce((sum, c) => sum + c.totalTimeMs, 0);
  const homeCluster = clusters.find(c => c.label === 'home');
  const workCluster = clusters.find(c => c.label === 'work');
  const homeTimePct = homeCluster && totalTime > 0 ? homeCluster.totalTimeMs / totalTime : 0;
  const workTimePct = workCluster && totalTime > 0 ? workCluster.totalTimeMs / totalTime : 0;

  // Count transitions (place changes in chronological order)
  let transitions = 0;
  if (visits.length > 1) {
    for (let i = 1; i < visits.length; i++) {
      if (visits[i].clusterIndex !== visits[i - 1].clusterIndex) {
        transitions++;
      }
    }
  }

  // Count novel places (not seen in weekly history)
  const weekClusters = computeLocationClusters(weeklyVisitHistory);
  let novelPlaces = 0;
  for (const cluster of clusters) {
    const isKnown = weekClusters.some(wc =>
      haversineDistance(cluster.latitude, cluster.longitude, wc.latitude, wc.longitude) < CLUSTER_RADIUS_METERS,
    );
    if (!isKnown) novelPlaces++;
  }

  // Diversity score
  const transitionBonus = transitions > 2 ? 10 : 0;
  const diversityScore = Math.min(100, (uniquePlaces - 1) * 25 + novelPlaces * 15 + transitionBonus);

  // Monotonous = only home and/or work
  const isMonotonous = uniquePlaces <= MONOTONOUS_THRESHOLD &&
    clusters.every(c => c.label === 'home' || c.label === 'work');

  return {
    date: today,
    uniquePlacesVisited: uniquePlaces,
    totalTransitions: transitions,
    diversityScore,
    homeTimePercent: homeTimePct,
    workTimePercent: workTimePct,
    novelPlaces,
    isMonotonous,
  };
}

// ============================================================
// Location Permission & Tracking
// ============================================================

/**
 * Request foreground location permission.
 * Returns true if granted.
 */
export async function requestLocationPermission(): Promise<boolean> {
  try {
    const { status } = await Location.requestForegroundPermissionsAsync();
    return status === 'granted';
  } catch {
    return false;
  }
}

/**
 * Start watching location changes.
 * Returns an unsubscribe function.
 */
export async function startLocationTracking(
  onLocation: (coords: { latitude: number; longitude: number; accuracy: number | null }) => void,
): Promise<{ remove: () => void } | null> {
  const granted = await requestLocationPermission();
  if (!granted) return null;

  const subscription = await Location.watchPositionAsync(
    {
      accuracy: Location.Accuracy.Balanced,
      distanceInterval: 50, // Only trigger on 50m+ movement
      timeInterval: 60000,  // At most once per minute
    },
    (location) => {
      onLocation({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        accuracy: location.coords.accuracy,
      });
    },
  );

  return subscription;
}

// ============================================================
// Mock Data (for development in Expo Go)
// ============================================================

/**
 * Generate plausible mock location diversity data.
 * Alternates between monotonous and diverse patterns.
 */
export function getMockLocationDiversity(): LocationDiversitySummary {
  const today = new Date().toISOString().slice(0, 10);
  const hour = new Date().getHours();

  // Simulate: morning at home, midday at work, sometimes a 3rd place
  const isAfternoon = hour >= 14;
  const hasThirdPlace = Math.random() > 0.5;

  const uniquePlaces = isAfternoon ? (hasThirdPlace ? 3 : 2) : (hour >= 9 ? 2 : 1);
  const diversityScore = Math.min(100, (uniquePlaces - 1) * 25 + (hasThirdPlace ? 15 : 0));

  return {
    date: today,
    uniquePlacesVisited: uniquePlaces,
    totalTransitions: Math.max(0, uniquePlaces - 1),
    diversityScore,
    homeTimePercent: uniquePlaces === 1 ? 1 : 0.45,
    workTimePercent: uniquePlaces >= 2 ? 0.4 : 0,
    novelPlaces: hasThirdPlace ? 1 : 0,
    isMonotonous: uniquePlaces <= 2 && !hasThirdPlace,
  };
}

/**
 * Generate mock weekly location diversity history.
 */
export function getMockWeeklyLocationDiversity(): { date: string; value: number }[] {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  return days.map(d => ({
    date: d,
    value: Math.round(15 + Math.random() * 55),
  }));
}
