import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Pose, POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';

// ═════════════════════════════════════════════════════════════════════
// POSE DEFINITIONS — All 7 Supported Asanas
// ═════════════════════════════════════════════════════════════════════
const POSES = [
  { id: 'auto',         label: 'Auto-Detect',  emoji: '🤖', sanskrit: 'AI Auto Detection' },
  { id: 'tadasana',     label: 'Tadasana',     emoji: '🧍', sanskrit: 'Mountain Pose' },
  { id: 'vrikshasana',  label: 'Vrikshasana',  emoji: '🌳', sanskrit: 'Tree Pose' },
  { id: 'bhujangasana', label: 'Bhujangasana', emoji: '🐍', sanskrit: 'Cobra Pose' },
  { id: 'trikonasana',  label: 'Trikonasana',  emoji: '📐', sanskrit: 'Triangle Pose' },
  { id: 'padmasana',    label: 'Padmasana',    emoji: '🪷', sanskrit: 'Lotus Pose' },
  { id: 'balasana',     label: 'Balasana',     emoji: '🧒', sanskrit: "Child's Pose" },
  { id: 'parvatasana',  label: 'Parvatasana',  emoji: '⛰️', sanskrit: 'Mountain Stretch' },
];

// ═════════════════════════════════════════════════════════════════════
// GEOMETRY UTILITIES — Enhanced precision helpers
// ═════════════════════════════════════════════════════════════════════

/** Angle at joint B formed by segments BA and BC (in degrees, 0-180) */
function calcAngle(a, b, c) {
  const rad = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let deg = Math.abs((rad * 180) / Math.PI);
  if (deg > 180) deg = 360 - deg;
  return deg;
}

/** Euclidean distance between two {x,y} landmarks */
function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/** Midpoint of two landmarks */
function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/**
 * Smooth linear scoring: returns a score between 0 and maxPts
 * based on how close `value` is to the ideal range [idealMin, idealMax].
 * Falls off linearly within the tolerance band.
 */
function gradedScore(value, idealMin, idealMax, tolerance, maxPts) {
  if (value >= idealMin && value <= idealMax) return maxPts;
  const gap = value < idealMin
    ? idealMin - value
    : value - idealMax;
  const ratio = Math.max(0, 1 - gap / tolerance);
  return Math.round(ratio * maxPts);
}

/** Check if a landmark is visible enough to be trusted */
function isVisible(lm, threshold = 0.5) {
  return (lm.visibility || 0) >= threshold;
}

/** Auto-detect pose from landmarks */
function detectPoseAutomatically(lm) {
  const coreVisible = [lm[11], lm[12], lm[23], lm[24], lm[25], lm[26], lm[27], lm[28]]
    .every(l => isVisible(l, 0.3));
  if (!coreVisible) return 'tadasana';

  const lSho = lm[11], rSho = lm[12];
  const lHip = lm[23], rHip = lm[24];
  const lKne = lm[25], rKne = lm[26];
  const lAnk = lm[27], rAnk = lm[28];
  const nose = lm[0];

  const shoMid = midpoint(lSho, rSho);
  const hipMid = midpoint(lHip, rHip);
  const ankMid = midpoint(lAnk, rAnk);
  const kneeMid = midpoint(lKne, rKne);

  const spineY = hipMid.y - shoMid.y; 
  const legY = ankMid.y - hipMid.y;   
  const headY = shoMid.y - nose.y;    

  if (hipMid.y < shoMid.y && hipMid.y < ankMid.y && spineY < -0.05) {
    return 'parvatasana';
  }

  const isLyingOrSeated = Math.abs(legY) < 0.25;

  if (isLyingOrSeated) {
    const chestLift = hipMid.y - shoMid.y; 
    const isChestLifted = chestLift > 0.1;
    const isFolded = chestLift < 0.1;

    if (isFolded && headY < -0.05) {
      return 'balasana';
    } else if (isChestLifted) {
      const kneeSpread = dist(lKne, rKne);
      const isSeated = Math.abs(hipMid.y - kneeMid.y) < 0.2;
      if (isSeated && kneeSpread > 0.2) {
        return 'padmasana';
      } else {
        return 'bhujangasana';
      }
    }
    return 'padmasana';
  } else {
    const feetSpread = dist(lAnk, rAnk);
    if (feetSpread > 0.3) {
      return 'trikonasana';
    } else {
      const lLegBend = calcAngle(lHip, lKne, lAnk);
      const rLegBend = calcAngle(rHip, rKne, rAnk);
      if (lLegBend < 150 || rLegBend < 150) {
         return 'vrikshasana';
      } else {
         return 'tadasana';
      }
    }
  }
}

// ═════════════════════════════════════════════════════════════════════
// ENHANCED POSE ANALYSIS ENGINE — All 7 Poses with Graded Scoring
// ═════════════════════════════════════════════════════════════════════
function analyzePose(poseName, lm) {
  /*  MediaPipe Pose Landmark Indices (33 total):
      0  nose         1  left_eye_inner    2  left_eye
      3  left_eye_outer  4  right_eye_inner  5  right_eye
      6  right_eye_outer 7  left_ear         8  right_ear
      9  mouth_left   10 mouth_right
      11 left_shoulder 12 right_shoulder
      13 left_elbow    14 right_elbow
      15 left_wrist    16 right_wrist
      17 left_pinky    18 right_pinky
      19 left_index    20 right_index
      21 left_thumb    22 right_thumb
      23 left_hip      24 right_hip
      25 left_knee     26 right_knee
      27 left_ankle    28 right_ankle
      29 left_heel     30 right_heel
      31 left_foot_index  32 right_foot_index
  */

  // Extract all key landmarks
  const nose = lm[0];
  const lSho = lm[11], rSho = lm[12];
  const lElb = lm[13], rElb = lm[14];
  const lWri = lm[15], rWri = lm[16];
  const lHip = lm[23], rHip = lm[24];
  const lKne = lm[25], rKne = lm[26];
  const lAnk = lm[27], rAnk = lm[28];
  const lHeel = lm[29], rHeel = lm[30];
  const lFoot = lm[31], rFoot = lm[32];

  // Common derived measurements
  const shoMid = midpoint(lSho, rSho);
  const hipMid = midpoint(lHip, rHip);
  const ankMid = midpoint(lAnk, rAnk);

  let totalScore = 0;
  const tips = [];
  const details = []; // internal detail tracking

  // Visibility-gated confidence: only score if key landmarks are visible
  const coreVisible = [lSho, rSho, lHip, rHip, lKne, rKne, lAnk, rAnk]
    .every(l => isVisible(l, 0.4));

  if (!coreVisible) {
    return {
      score: 0,
      tips: ['⚠️ Cannot detect your full body. Please step back so your entire body is visible in the frame.']
    };
  }

  switch (poseName) {

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 1. TADASANA — Mountain Pose
    //    Criteria: Vertical spine, straight legs, arms at sides,
    //              level shoulders, feet together, weight centered
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'tadasana': {
      // (1) Spine vertical alignment — shoulder-hip X offset (max 20 pts)
      const spineOffset = Math.abs(shoMid.x - hipMid.x);
      const spineScore = gradedScore(spineOffset, 0, 0.03, 0.08, 20);
      totalScore += spineScore;
      if (spineScore >= 18) tips.push('✅ Spine is perfectly vertical');
      else if (spineScore >= 10) tips.push('⚠️ Slight lean detected — align head over hips');
      else tips.push('❌ Your torso is leaning — stand straight, weight centered');

      // (2) Left leg straight: hip-knee-ankle angle ~180° (max 15 pts)
      const lLeg = calcAngle(lHip, lKne, lAnk);
      const lLegScore = gradedScore(lLeg, 168, 180, 20, 15);
      totalScore += lLegScore;

      // (3) Right leg straight (max 15 pts)
      const rLeg = calcAngle(rHip, rKne, rAnk);
      const rLegScore = gradedScore(rLeg, 168, 180, 20, 15);
      totalScore += rLegScore;

      if (lLegScore + rLegScore >= 26) tips.push('✅ Both legs are straight and firm');
      else if (lLegScore + rLegScore >= 15) tips.push('⚠️ Straighten your legs more — engage your quads');
      else tips.push('❌ Bend detected in legs — fully extend both knees');

      // (4) Arms at sides — wrist close to hips (max 15 pts)
      const lArmD = dist(lWri, lHip);
      const rArmD = dist(rWri, rHip);
      const avgArmD = (lArmD + rArmD) / 2;
      const armScore = gradedScore(avgArmD, 0, 0.12, 0.15, 15);
      totalScore += armScore;
      if (armScore >= 12) tips.push('✅ Arms relaxed by your sides');
      else tips.push('⚠️ Let your arms hang naturally beside your hips');

      // (5) Shoulders level (max 15 pts)
      const shoulderTilt = Math.abs(lSho.y - rSho.y);
      const shldrScore = gradedScore(shoulderTilt, 0, 0.02, 0.06, 15);
      totalScore += shldrScore;
      if (shldrScore >= 12) tips.push('✅ Shoulders are level and relaxed');
      else tips.push('⚠️ One shoulder is higher — relax and level them');

      // (6) Feet together (max 10 pts)
      const feetDist = dist(lAnk, rAnk);
      const feetScore = gradedScore(feetDist, 0, 0.08, 0.15, 10);
      totalScore += feetScore;
      if (feetScore >= 8) tips.push('✅ Feet are together');
      else tips.push('⚠️ Bring your feet closer together');

      // (7) Head aligned over hips (max 10 pts)
      const headOffset = Math.abs(nose.x - hipMid.x);
      const headScore = gradedScore(headOffset, 0, 0.04, 0.08, 10);
      totalScore += headScore;
      if (headScore < 6) tips.push('⚠️ Center your head over your hips — gaze forward');
      break;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 2. VRIKSHASANA — Tree Pose
    //    Criteria: One leg bent with foot on inner thigh (above knee),
    //              standing leg straight, arms overhead, spine vertical,
    //              hips level, stability (low sway)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'vrikshasana': {
      const lLegAng = calcAngle(lHip, lKne, lAnk);
      const rLegAng = calcAngle(rHip, rKne, rAnk);
      const leftBent = lLegAng < 145;
      const rightBent = rLegAng < 145;

      // (1) One leg must be bent (max 20 pts)
      if (!leftBent && !rightBent) {
        totalScore += 5;
        tips.push('❌ Bend one leg – place your foot on the inner thigh of the standing leg');
      } else {
        totalScore += 20;
        const bentAnkle = leftBent ? lAnk : rAnk;
        const bentKnee = leftBent ? lKne : rKne;
        const standKnee = leftBent ? rKne : lKne;
        const standLeg = leftBent ? rLegAng : lLegAng;

        // (2) Foot above standing knee (max 20 pts)
        if (bentAnkle.y < standKnee.y - 0.02) {
          totalScore += 20;
          tips.push('✅ Foot placed high on inner thigh – excellent');
        } else if (bentAnkle.y < standKnee.y + 0.05) {
          totalScore += 10;
          tips.push('⚠️ Foot is near knee level – lift it higher above the knee');
        } else {
          tips.push('❌ Foot is too low – place it on the inner thigh, NOT the knee');
        }

        // (3) Bent knee pointing outward (max 10 pts)
        const bentKneeOutward = leftBent
          ? (bentKnee.x < lHip.x - 0.03)
          : (bentKnee.x > rHip.x + 0.03);
        if (bentKneeOutward) {
          totalScore += 10;
          tips.push('✅ Bent knee is opening outward');
        } else {
          tips.push('⚠️ Rotate your bent knee outward to open the hip');
        }

        // (4) Standing leg straight (max 15 pts)
        const standScore = gradedScore(standLeg, 168, 180, 18, 15);
        totalScore += standScore;
        if (standScore >= 12) tips.push('✅ Standing leg is firm and straight');
        else tips.push('⚠️ Fully extend your standing leg – lock the knee gently');
      }

      // (5) Arms raised above head (max 15 pts)
      const armsUp = (lWri.y < lSho.y - 0.05) && (rWri.y < rSho.y - 0.05);
      const handsClose = dist(lWri, rWri) < 0.15;
      if (armsUp && handsClose) { totalScore += 15; tips.push('✅ Arms raised with palms together – Namaste!'); }
      else if (armsUp) { totalScore += 10; tips.push('⚠️ Arms are up – bring your palms together'); }
      else tips.push('❌ Raise both arms above your head');

      // (6) Spine vertical (max 10 pts)
      const spineOff = Math.abs(shoMid.x - hipMid.x);
      const spineS = gradedScore(spineOff, 0, 0.04, 0.08, 10);
      totalScore += spineS;
      if (spineS < 6) tips.push('⚠️ Keep your torso upright – avoid leaning');

      // (7) Hips level (max 10 pts)
      const hipTilt = Math.abs(lHip.y - rHip.y);
      const hipScore = gradedScore(hipTilt, 0, 0.02, 0.06, 10);
      totalScore += hipScore;
      if (hipScore < 6) tips.push('⚠️ Level your hips – the lifted side is dropping');
      break;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 3. BHUJANGASANA — Cobra Pose
    //    Criteria: Chest lifted above hips, elbows extended (not locked),
    //              hips grounded close to ankles, head/gaze up,
    //              shoulders down (not shrugged), symmetry
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'bhujangasana': {
      // (1) Chest lift — shoulders well above hips (max 25 pts)
      const chestLift = hipMid.y - shoMid.y;
      const liftScore = gradedScore(chestLift, 0.05, 0.4, 0.08, 25);
      totalScore += liftScore;
      if (liftScore >= 20) tips.push('✅ Excellent chest lift');
      else if (liftScore >= 10) tips.push('⚠️ Lift your chest higher – press through your palms');
      else tips.push('❌ Your chest is too low – arch your upper back upward');

      // (2) Elbow angles — should be 130-170° (max 20 pts)
      const lElbA = calcAngle(lSho, lElb, lWri);
      const rElbA = calcAngle(rSho, rElb, rWri);
      const avgElb = (lElbA + rElbA) / 2;
      const elbScore = gradedScore(avgElb, 130, 170, 25, 20);
      totalScore += elbScore;
      if (elbScore >= 16) tips.push('✅ Arms supporting body correctly');
      else if (avgElb < 130) tips.push('⚠️ Extend your arms more – push up through palms');
      else tips.push("⚠️ Ease your arms back slightly – don't lock elbows");

      // (3) Elbow symmetry (max 5 pts)
      const elbDiff = Math.abs(lElbA - rElbA);
      const elbSymScore = gradedScore(elbDiff, 0, 10, 25, 5);
      totalScore += elbSymScore;
      if (elbSymScore < 3) tips.push('⚠️ Even out your arms – one elbow is bent more');

      // (4) Hips grounded — hips close to ankle Y level (max 20 pts)
      const hipAnkDiff = Math.abs(hipMid.y - ankMid.y);
      const hipGroundScore = gradedScore(hipAnkDiff, 0, 0.12, 0.15, 20);
      totalScore += hipGroundScore;
      if (hipGroundScore >= 16) tips.push('✅ Hips and legs grounded on the floor');
      else tips.push('⚠️ Press your hips and thighs into the floor');

      // (5) Head up / gaze forward — nose above shoulders (max 15 pts)
      const headUp = shoMid.y - nose.y;
      const headScore = gradedScore(headUp, 0.03, 0.3, 0.06, 15);
      totalScore += headScore;
      if (headScore >= 12) tips.push('✅ Head lifted – gaze forward and upward');
      else tips.push('⚠️ Lift your head – look forward or slightly upward');

      // (6) Shoulders away from ears (max 15 pts)
      const shoulderEarDist = Math.abs(shoMid.y - nose.y);
      const neckL = gradedScore(shoulderEarDist, 0.08, 0.3, 0.06, 15);
      totalScore += neckL;
      if (neckL < 8) tips.push('⚠️ Roll your shoulders back and down – away from ears');
      break;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 4. TRIKONASANA — Triangle Pose
    //    Criteria: Wide stance, both legs straight, lateral bend with
    //              one hand down one up, shoulders vertically stacked,
    //              torso in a plane (not rotating forward/back)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'trikonasana': {
      // (1) Wide foot stance (max 15 pts)
      const feetSpread = dist(lAnk, rAnk);
      const feetScore = gradedScore(feetSpread, 0.30, 0.7, 0.12, 15);
      totalScore += feetScore;
      if (feetScore >= 12) tips.push('✅ Excellent wide stance');
      else tips.push('⚠️ Widen your feet more – step apart');

      // (2) Left leg straight (max 12 pts)
      const lLeg = calcAngle(lHip, lKne, lAnk);
      const lLS = gradedScore(lLeg, 162, 180, 18, 12);
      totalScore += lLS;

      // (3) Right leg straight (max 12 pts)
      const rLeg = calcAngle(rHip, rKne, rAnk);
      const rLS = gradedScore(rLeg, 162, 180, 18, 12);
      totalScore += rLS;
      if (lLS + rLS >= 20) tips.push('✅ Both legs are straight');
      else tips.push('⚠️ Straighten both legs fully – engage your thigh muscles');

      // (4) Arm triangle — one wrist below hips, one above shoulders (max 25 pts)
      const lDown = lWri.y > lHip.y + 0.03;
      const rDown = rWri.y > rHip.y + 0.03;
      const lUp = lWri.y < lSho.y - 0.05;
      const rUp = rWri.y < rSho.y - 0.05;
      if ((lDown && rUp) || (rDown && lUp)) {
        totalScore += 25;
        tips.push('✅ Arms in perfect triangle position');
      } else if ((lWri.y > lHip.y || rWri.y > rHip.y)) {
        totalScore += 12;
        tips.push('⚠️ Extend the upper arm straight to the ceiling');
      } else {
        tips.push('❌ Reach one hand toward your ankle, the other straight up');
      }

      // (5) Lateral shoulder tilt — shoulders should be stacked vertically (max 20 pts)
      const shoYDiff = Math.abs(lSho.y - rSho.y);
      const latScore = gradedScore(shoYDiff, 0.08, 0.5, 0.08, 20);
      totalScore += latScore;
      if (latScore >= 16) tips.push('✅ Deep lateral bend – shoulders stacked');
      else tips.push('⚠️ Bend deeper sideways – stack shoulders vertically');

      // (6) Torso alignment — hips shouldn't rotate (max 8 pts)
      const hipXDiff = Math.abs(lHip.x - rHip.x);
      if (hipXDiff > 0.05) { totalScore += 8; tips.push('✅ Hips open — good alignment'); }
      else tips.push("⚠️ Open your hips – don't let your torso rotate forward");

      // (7) Head turned toward upper hand (max 8 pts)
      const upperWrist = (lDown) ? rWri : lWri;
      if (nose.y < shoMid.y) { totalScore += 8; }
      else tips.push('⚠️ Turn your gaze upward toward your raised hand');
      break;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 5. PADMASANA — Lotus Pose
    //    Criteria: Seated, cross-legged, knees apart, spine erect,
    //              hands on knees, shoulders relaxed, head centered
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'padmasana': {
      // (1) Seated — hips & knees at similar Y (max 20 pts)
      const hipKneeYDiff = Math.abs(hipMid.y - ((lKne.y + rKne.y) / 2));
      const seatedScore = gradedScore(hipKneeYDiff, 0, 0.12, 0.15, 20);
      totalScore += seatedScore;
      if (seatedScore >= 16) tips.push('✅ Seated position detected');
      else tips.push('⚠️ Sit down on the floor in a cross-legged position');

      // (2) Knees spread outward (max 20 pts)
      const kneeSpread = dist(lKne, rKne);
      const kneeScore = gradedScore(kneeSpread, 0.18, 0.6, 0.1, 20);
      totalScore += kneeScore;
      if (kneeScore >= 16) tips.push('✅ Knees are open in lotus position');
      else tips.push('⚠️ Open your knees outward – let them drop naturally');

      // (3) Spine erect (max 20 pts)
      const spineOff = Math.abs(shoMid.x - hipMid.x);
      const spineScore = gradedScore(spineOff, 0, 0.04, 0.08, 20);
      totalScore += spineScore;
      if (spineScore >= 16) tips.push('✅ Spine is beautifully erect');
      else if (spineScore >= 8) tips.push('⚠️ Straighten your back — sit taller');
      else tips.push('❌ Your back is slouched – elongate your spine upward');

      // (4) Hands on or near knees (max 15 pts)
      const lHK = dist(lWri, lKne);
      const rHK = dist(rWri, rKne);
      const avgHK = (lHK + rHK) / 2;
      const handScore = gradedScore(avgHK, 0, 0.12, 0.15, 15);
      totalScore += handScore;
      if (handScore >= 12) tips.push('✅ Hands resting on knees');
      else tips.push('⚠️ Place your hands gently on your knees, palms down or in mudra');

      // (5) Shoulders relaxed and level (max 15 pts)
      const shldrTilt = Math.abs(lSho.y - rSho.y);
      const shldrScore = gradedScore(shldrTilt, 0, 0.02, 0.06, 15);
      totalScore += shldrScore;
      if (shldrScore < 8) tips.push('⚠️ Relax your shoulders – let them drop evenly');

      // (6) Head centered over spine (max 10 pts)
      const headOff = Math.abs(nose.x - hipMid.x);
      const headS = gradedScore(headOff, 0, 0.04, 0.08, 10);
      totalScore += headS;
      if (headS < 6) tips.push('⚠️ Center your head – gaze straight ahead');
      break;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 6. BALASANA — Child's Pose
    //    Criteria: Torso folded forward, knees deeply bent, arms forward
    //              or by sides, forehead toward floor, hips toward heels
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'balasana': {
      // (1) Torso folded — shoulders at or below hip level (max 25 pts)
      const foldDiff = shoMid.y - hipMid.y; // positive = folded
      const foldScore = gradedScore(foldDiff, -0.03, 0.4, 0.1, 25);
      totalScore += foldScore;
      if (foldScore >= 20) tips.push('✅ Torso is folded forward — great');
      else if (foldScore >= 10) tips.push('⚠️ Fold deeper – bring your chest toward your thighs');
      else tips.push('❌ You need to fold forward – hinge at the hips');

      // (2) Left knee deeply bent (max 12 pts)
      const lKA = calcAngle(lHip, lKne, lAnk);
      const lKS = gradedScore(lKA, 0, 80, 40, 12);
      totalScore += lKS;

      // (3) Right knee deeply bent (max 12 pts)
      const rKA = calcAngle(rHip, rKne, rAnk);
      const rKS = gradedScore(rKA, 0, 80, 40, 12);
      totalScore += rKS;
      if (lKS + rKS >= 20) tips.push('✅ Knees tucked under body');
      else if (lKS + rKS >= 10) tips.push('⚠️ Bend your knees more – sit back on your heels');
      else tips.push('❌ Kneel down fully – sit your hips back onto your heels');

      // (4) Hips close to heels (max 15 pts)
      const hipHeelDist = (dist(lHip, lAnk) + dist(rHip, rAnk)) / 2;
      const hhScore = gradedScore(hipHeelDist, 0, 0.2, 0.15, 15);
      totalScore += hhScore;
      if (hhScore >= 12) tips.push('✅ Hips resting on heels');
      else tips.push('⚠️ Sink your hips back toward your heels');

      // (5) Arms extended forward or along sides (max 18 pts)
      const armsForward = lWri.y > lSho.y - 0.05 || rWri.y > rSho.y - 0.05;
      if (armsForward) { totalScore += 18; tips.push('✅ Arms positioned well'); }
      else { totalScore += 5; tips.push('⚠️ Extend your arms forward or rest them alongside your body'); }

      // (6) Forehead down — nose close to or below shoulder level (max 18 pts)
      const headDown = nose.y - shoMid.y;
      const headScore = gradedScore(headDown, -0.05, 0.3, 0.08, 18);
      totalScore += headScore;
      if (headScore >= 14) tips.push('✅ Head is resting toward the floor');
      else tips.push('⚠️ Lower your forehead toward the ground');
      break;
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 7. PARVATASANA — Mountain Stretch / Downward-Facing Dog
    //    Criteria: Hips highest (inverted V), arms straight, legs straight,
    //              head between arms, equal weight distribution
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    case 'parvatasana': {
      // (1) Hips are the highest point — above shoulders AND ankles (max 25 pts)
      const hipAboveSho = shoMid.y - hipMid.y;
      const hipAboveAnk = ankMid.y - hipMid.y;
      if (hipAboveSho > 0.03 && hipAboveAnk > 0.03) {
        totalScore += 25;
        tips.push('✅ Hips are the highest point – beautiful inverted V');
      } else if (hipAboveSho > 0 || hipAboveAnk > 0) {
        totalScore += 12;
        tips.push('⚠️ Push your hips higher – tailbone toward the ceiling');
      } else {
        tips.push('❌ Your hips need to be the highest point – form an inverted V');
      }

      // (2) Left arm straight (max 12 pts)
      const lArmA = calcAngle(lSho, lElb, lWri);
      const lAS = gradedScore(lArmA, 160, 180, 20, 12);
      totalScore += lAS;

      // (3) Right arm straight (max 12 pts)
      const rArmA = calcAngle(rSho, rElb, rWri);
      const rAS = gradedScore(rArmA, 160, 180, 20, 12);
      totalScore += rAS;
      if (lAS + rAS >= 20) tips.push('✅ Arms fully extended');
      else tips.push('⚠️ Straighten your arms – push the floor away');

      // (4) Left leg straight (max 10 pts)
      const lLeg = calcAngle(lHip, lKne, lAnk);
      const lLS = gradedScore(lLeg, 158, 180, 20, 10);
      totalScore += lLS;

      // (5) Right leg straight (max 10 pts)
      const rLeg = calcAngle(rHip, rKne, rAnk);
      const rLS = gradedScore(rLeg, 158, 180, 20, 10);
      totalScore += rLS;
      if (lLS + rLS >= 16) tips.push('✅ Legs are straight — heels driving down');
      else tips.push('⚠️ Straighten your legs – push heels toward the floor');

      // (6) Head between arms — nose Y below or at shoulder Y (max 15 pts)
      const headBetween = nose.y - shoMid.y;
      const hbScore = gradedScore(headBetween, 0.02, 0.3, 0.06, 15);
      totalScore += hbScore;
      if (hbScore >= 12) tips.push('✅ Head relaxed between arms');
      else tips.push('⚠️ Let your head hang naturally – look at your navel');

      // (7) Arm-leg symmetry (max 8 pts)
      const armDiff = Math.abs(lArmA - rArmA);
      const legDiff = Math.abs(lLeg - rLeg);
      const symScore = gradedScore(armDiff + legDiff, 0, 12, 30, 8);
      totalScore += symScore;
      if (symScore < 4) tips.push('⚠️ Try to be more symmetrical – balance evenly on both sides');

      // (8) Shoulder width — not too narrow (max 8 pts)
      const shoWidth = dist(lSho, rSho);
      if (shoWidth > 0.1) { totalScore += 8; }
      else tips.push('⚠️ Keep your hands shoulder-width apart');
      break;
    }

    default:
      tips.push('Select a yoga pose to begin real-time analysis');
      break;
  }

  return { score: Math.min(Math.max(totalScore, 0), 100), tips };
}

// ═════════════════════════════════════════════════════════════════════
// POSE DESCRIPTIONS
// ═════════════════════════════════════════════════════════════════════
const POSE_DESCRIPTIONS = {
  tadasana: 'Stand with feet together, spine straight, arms relaxed by your sides. The foundation of all standing poses. Focus on grounding through all four corners of your feet.',
  vrikshasana: 'Stand on one leg, place the other foot high on your inner thigh (never on the knee), raise arms overhead with palms together. Focus on a fixed point for balance.',
  bhujangasana: 'Lie prone, press palms down by your chest, lift your upper body by arching the spine. Keep hips, thighs, and feet on the floor. Gaze forward and up.',
  trikonasana: 'Stand with feet wide apart (~4 feet), extend arms, bend sideways to touch one ankle while the other arm reaches straight up. Both legs stay straight.',
  padmasana: 'Sit cross-legged in lotus position, each foot resting on the opposite thigh. Spine erect, hands on knees in a mudra. Foundation for meditation and pranayama.',
  balasana: 'Kneel on the floor, sit back on your heels, then fold forward with forehead toward the ground. Arms extended forward or alongside the body. A resting pose.',
  parvatasana: 'From all fours, push your hips up and back to form an inverted V shape. Arms and legs are straight, head relaxes between the upper arms. Press heels toward the floor.',
  auto: 'AI Agent will automatically analyze your posture and detect the asana you are performing from the list of supported poses.',
};

// ═════════════════════════════════════════════════════════════════════
// MAIN APP COMPONENT
// ═════════════════════════════════════════════════════════════════════
function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [activePose, setActivePose] = useState('auto');
  const [score, setScore] = useState(0);
  const [tips, setTips] = useState([]);
  const [loading, setLoading] = useState(true);
  const [cameraReady, setCameraReady] = useState(false);
  const [sessionTime, setSessionTime] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const landmarkBuffer = useRef([]);   // sequential smoothing buffer (15 frames)
  const sessionTimer = useRef(null);

  // Start session timer
  useEffect(() => {
    sessionTimer.current = setInterval(() => setSessionTime(t => t + 1), 1000);
    return () => clearInterval(sessionTimer.current);
  }, []);

  // Track best score
  useEffect(() => {
    if (score > bestScore) setBestScore(score);
  }, [score]);

  // Reset best score on pose change
  useEffect(() => {
    setBestScore(0);
    setScore(0);
    setTips([]);
  }, [activePose]);

  const formatTime = (s) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  // ── Process results from MediaPipe ─────────────────────────────────
  const onResults = useCallback((results) => {
    if (!canvasRef.current) return;
    setLoading(false);

    const ctx = canvasRef.current.getContext('2d');
    const w = canvasRef.current.width;
    const h = canvasRef.current.height;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(results.image, 0, 0, w, h);

    if (results.poseLandmarks) {
      // ── Draw skeleton with colour-coded confidence ──
      drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
        color: 'rgba(192,132,252,0.7)', lineWidth: 3
      });
      drawLandmarks(ctx, results.poseLandmarks, {
        color: '#2dd4bf', fillColor: 'rgba(45,212,191,0.6)', lineWidth: 1, radius: 4
      });

      // ── Sequential smoothing: keep last 15 frames ──
      landmarkBuffer.current.push(
        results.poseLandmarks.map(l => ({ x: l.x, y: l.y, visibility: l.visibility }))
      );
      if (landmarkBuffer.current.length > 15) landmarkBuffer.current.shift();

      // Weighted average — more recent frames have higher weight
      const bufLen = landmarkBuffer.current.length;
      const avgLm = results.poseLandmarks.map((lm, idx) => {
        let totalW = 0, sx = 0, sy = 0, sv = 0;
        for (let fi = 0; fi < bufLen; fi++) {
          const weight = (fi + 1); // linear weighting: older=1, newest=bufLen
          const pt = landmarkBuffer.current[fi][idx];
          sx += pt.x * weight;
          sy += pt.y * weight;
          sv += (pt.visibility || 0) * weight;
          totalW += weight;
        }
        return { x: sx / totalW, y: sy / totalW, visibility: sv / totalW };
      });

      let detectedPose = activePose;
      let isAuto = false;
      if (activePose === 'auto') {
        detectedPose = detectPoseAutomatically(avgLm);
        isAuto = true;
      }

      const result = analyzePose(detectedPose, avgLm);
      if (isAuto) {
         const autoLabel = POSES.find(p => p.id === detectedPose)?.label || detectedPose;
         result.tips.unshift(`🤖 Auto-Detected Pose: ${autoLabel}`);
      }
      
      setScore(result.score);
      setTips(result.tips);
    }
    ctx.restore();
  }, [activePose]);

  // ── Initialize MediaPipe + Camera ──────────────────────────────────
  useEffect(() => {
    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    pose.setOptions({
      modelComplexity: 2,           // ENHANCED: Full model for maximum accuracy
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.6,   // ENHANCED: Higher threshold
      minTrackingConfidence: 0.6,    // ENHANCED: Higher threshold
    });
    pose.onResults(onResults);

    if (videoRef.current) {
      const camera = new Camera(videoRef.current, {
        onFrame: async () => { await pose.send({ image: videoRef.current }); },
        width: 1280,
        height: 720,
      });
      camera.start().then(() => setCameraReady(true));
    }
  }, [onResults]);

  // ── Score colour helper ────────────────────────────────────────────
  const scoreColor = score >= 80 ? '#2dd4bf' : score >= 50 ? '#fbbf24' : '#f87171';
  const currentPoseInfo = POSES.find(p => p.id === activePose);

  return (
    <div className="app-shell">
      {/* ========== NAVBAR ========== */}
      <nav className="navbar">
        <div className="nav-brand">
          <span className="brand-icon">🧘</span>
          <span className="brand-text">ZenSense AI</span>
        </div>
        <div className="nav-right">
          <div className="nav-timer">{formatTime(sessionTime)}</div>
          <div className="nav-status">
            <span className={`status-dot ${loading ? 'loading' : 'ready'}`}></span>
            {loading ? 'Initializing…' : 'Engine Active'}
          </div>
        </div>
      </nav>

      {/* ========== MAIN LAYOUT ========== */}
      <div className="main-layout">
        {/* ── Left: Camera ────────────────── */}
        <div className="camera-section">
          <div className="camera-wrapper">
            <video className="hidden-video" ref={videoRef} playsInline muted></video>
            <canvas ref={canvasRef} width={1280} height={720} className="camera-canvas"></canvas>

            {/* Floating overlays */}
            <div className="cam-overlay-top">
              <div className="cam-badge live-badge">● LIVE</div>
              <div className="cam-badge pose-badge">{currentPoseInfo?.emoji} {currentPoseInfo?.label}</div>
              <div className="cam-badge model-badge">Model v2 • Complexity 2</div>
            </div>

            <div className="cam-overlay-bottom">
              <div className="cam-badge best-badge">Best: {bestScore}%</div>
              <div className="cam-score-pill" style={{ borderColor: scoreColor }}>
                <span className="cam-score-num" style={{ color: scoreColor }}>{score}%</span>
                <span className="cam-score-label">Accuracy</span>
              </div>
            </div>
          </div>

          {/* ── Insights bar below camera ─── */}
          <div className="insights-bar">
            <div className="insight-item">
              <span className="insight-icon">🦴</span>
              <div>
                <strong>Joint Identification (33 Landmarks)</strong>
                <p>Each frame maps 33 skeletal landmarks to create an invariant representation immune to lighting, background, and clothing. Precise angle calculations enable biomechanical analysis — e.g., measuring 168°+ knee extension in Tadasana.</p>
              </div>
            </div>
            <div className="insight-item">
              <span className="insight-icon">🎬</span>
              <div>
                <strong>Sequential Temporal Smoothing (15-frame)</strong>
                <p>A weighted sliding window averages landmarks over 15 frames with recency bias (newer frames weighted higher). This eliminates sensor jitter and enables hold-stability measurement — critical for yoga's sustained postures.</p>
              </div>
            </div>
          </div>
        </div>

        {/* ── Right: Sidebar ──────────────── */}
        <aside className="sidebar">
          {/* Score Card */}
          <div className="sidebar-card score-card">
            <h2 className="card-label">Posture Score</h2>
            <div className="big-score" style={{ color: scoreColor }}>{score}<span className="score-pct">%</span></div>
            <div className="score-bar-track">
              <div className="score-bar-fill" style={{ width: `${score}%`, background: scoreColor }}></div>
            </div>
            <p className="score-verdict">
              {score >= 90 ? '🏆 Perfect form! Hold it steady.' :
               score >= 75 ? '🎉 Great posture! Minor adjustments below.' :
               score >= 50 ? '💪 Getting there — follow the corrections.' :
               score >= 25 ? '🧐 Needs work — focus on the key corrections.' :
               '⏳ Position yourself for the selected pose.'}
            </p>
          </div>

          {/* Live Feedback */}
          <div className="sidebar-card feedback-card">
            <h3 className="card-label">Live Corrections ({tips.filter(t => t.startsWith('✅')).length}/{tips.length} checks passed)</h3>
            <ul className="tip-list">
              {tips.map((t, i) => (
                <li key={i} className={`tip-item ${t.startsWith('✅') ? 'tip-pass' : t.startsWith('❌') ? 'tip-fail' : 'tip-warn'}`}>{t}</li>
              ))}
              {tips.length === 0 && <li className="tip-item">Waiting for pose detection…</li>}
            </ul>
          </div>

          {/* Pose selector */}
          <div className="sidebar-card select-card">
            <h3 className="card-label">Select Asana (7 Supported)</h3>
            <div className="pose-grid">
              {POSES.map(p => (
                <button
                  key={p.id}
                  className={`pose-btn ${activePose === p.id ? 'active' : ''}`}
                  onClick={() => { setActivePose(p.id); landmarkBuffer.current = []; }}
                >
                  <span className="pose-btn-emoji">{p.emoji}</span>
                  <span className="pose-btn-name">{p.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Pose description */}
          <div className="sidebar-card desc-card">
            <h3 className="card-label">{currentPoseInfo?.emoji} {currentPoseInfo?.label} <span style={{ fontWeight: 300, fontSize: '0.8rem' }}>({currentPoseInfo?.sanskrit})</span></h3>
            <p className="desc-text">{POSE_DESCRIPTIONS[activePose]}</p>
          </div>
        </aside>
      </div>

      {/* ========== FOOTER ========== */}
      <footer className="app-footer">
        ZenSense AI v2.0 • Enhanced Geometric Analysis • 33-Point Skeleton • 15-Frame Temporal Smoothing • 7 Asanas • MediaPipe Pose (Full Model)
      </footer>
    </div>
  );
}

export default App;
