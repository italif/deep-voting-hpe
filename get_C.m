function [C] = get_C()
C.r_ankle = 0;
C.r_knee = 1;
C.r_hip = 2;
C.l_hip = 3;
C.l_knee = 4;
C.l_ankle = 5;
C.pelvis = 6;
C.thorax = 7;
C.upper_neck = 8;
C.head_top = 9;
C.r_wrist = 10;
C.r_elbow = 11;
C.r_shoulder = 12;
C.l_shoulder = 13;
C.l_elbow = 14;
C.l_wrist = 15;

C.r_hand = 16;
C.l_hand = 17;

C.r_low_leg = 18;
C.r_up_leg = 19;
C.l_up_leg = 20;
C.l_low_leg = 21;
C.mid_body = 22;
C.r_body = 23;
C.l_body = 24;
C.head_center = 25;
C.r_low_arm = 26;
C.r_up_arm = 27;
C.l_up_arm = 28;
C.l_low_arm = 29;

C.N_pts = 18; %number of points
C.N_dense_pts = 30; %number of dense points

C.upper_body = [
    C.upper_neck;
    C.head_top;
    C.r_wrist;
    C.r_elbow;
    C.r_shoulder;
    C.l_shoulder;
    C.l_elbow;
    C.l_wrist;
    C.r_hand;
    C.l_hand;
    ];

C.Pts_indx_list = {
    C.r_ankle,'r_ankle';
    C.r_knee,'r_knee';
    C.r_hip,'r_hip';
    C.l_hip,'l_hip';
    C.l_knee,'l_knee';
    C.l_ankle,'l_ankle';
    C.pelvis,'pelvis';
    C.thorax,'thorax';
    C.upper_neck,'upper_neck';
    C.head_top,'head_top';
    C.r_wrist,'r_wrist';
    C.r_elbow,'r_elbow';
    C.r_shoulder,'r_shoulder';
    C.l_shoulder,'l_shoulder';
    C.l_elbow,'l_elbow';
    C.l_wrist,'l_wrist';
    C.r_hand,'r_hand';
    C.l_hand,'l_hand';
    
    C.r_low_leg,'r_low_leg';
    C.r_up_leg,'r_up_leg';
    C.l_up_leg,'l_up_leg';
    C.l_low_leg,'l_low_leg';
    C.mid_body,'mid_body';
    C.r_body,'r_body';
    C.l_body,'l_body';
    C.head_center,'head_center';
    C.r_low_arm,'r_low_arm';
    C.r_up_arm,'r_up_arm';
    C.l_up_arm,'l_up_arm';
    C.l_low_arm,'l_low_arm'};
C.Pts_list = C.Pts_indx_list(:,2);

C.Flip_map = [...
    C.r_ankle, C.l_ankle;
    C.r_knee, C.l_knee;
    C.r_hip, C.l_hip;
    C.l_hip, C.r_hip;
    C.l_knee, C.r_knee;
    C.l_ankle, C.r_ankle;
    C.pelvis, C.pelvis;
    C.thorax, C.thorax;
    C.upper_neck, C.upper_neck;
    C.head_top, C.head_top;
    C.r_wrist, C.l_wrist;
    C.r_elbow, C.l_elbow;
    C.r_shoulder, C.l_shoulder;
    C.l_shoulder, C.r_shoulder;
    C.l_elbow, C.r_elbow;
    C.l_wrist, C.r_wrist;
    C.r_hand, C.l_hand;
    C.l_hand, C.r_hand];

C.Dense_map = [...
    C.r_ankle, C.r_knee;
    C.r_knee, C.r_hip;
    C.l_knee, C.l_hip;
    C.l_ankle, C.l_knee;
    C.pelvis, C.thorax;
    C.r_hip, C.r_shoulder;
    C.l_hip, C.l_shoulder;
    C.head_top, C.upper_neck;
    C.r_wrist, C.r_elbow;
    C.r_elbow, C.r_shoulder;
    C.l_elbow, C.l_shoulder;
    C.l_wrist, C.l_elbow];

C.CRF_pairs_ex4_r_b = [...
    C.head_top, C.head_center, 36, 1;
    C.head_center, C.upper_neck, 36, 1;
    C.head_top, C.upper_neck, 72, 0;
    
    C.r_shoulder, C.r_up_arm, 84, 1;
    C.r_up_arm, C.r_elbow, 84, 1;
    C.r_shoulder, C.r_elbow, 168, 0;
    
    C.r_elbow, C.r_low_arm, 84, 1;
    C.r_low_arm, C.r_wrist, 84, 1;
    C.r_elbow, C.r_wrist, 168, 0;
    
    
    C.l_shoulder, C.l_up_arm, 84, 1;
    C.l_up_arm, C.l_elbow, 84, 1;
    C.l_shoulder, C.l_elbow, 168, 0;
    
    C.l_elbow, C.l_low_arm, 84, 1;
    C.l_low_arm, C.l_wrist, 84, 1;
    C.l_elbow, C.l_wrist, 168, 0;
    
    C.thorax, C.mid_body, 132, 1;
    C.mid_body, C.pelvis, 132, 1;
    C.thorax, C.pelvis, 264, 0;
    
    
    C.r_hip, C.r_up_leg, 117, 1;
    C.r_up_leg, C.r_knee, 117, 1;
    C.r_hip, C.r_knee, 234, 0;
    
    C.r_knee, C.r_low_leg, 111, 1;
    C.r_low_leg, C.r_ankle, 111, 1;
    C.r_knee, C.r_ankle, 222, 0;
    
    C.l_hip, C.l_up_leg, 117, 1;
    C.l_up_leg, C.l_knee, 117, 1;
    C.l_hip, C.l_knee, 234, 0;
    
    C.l_knee, C.l_low_leg, 111, 1;
    C.l_low_leg, C.l_ankle, 111, 1;
    C.l_knee, C.l_ankle, 222, 0;
    
    C.r_shoulder, C.r_body, 132, 1;
    C.r_body, C.r_hip, 132, 1;
    C.r_shoulder, C.r_hip, 264, 0;
    
    C.l_shoulder, C.l_body, 132, 1;
    C.l_body, C.l_hip, 132, 1;
    C.l_shoulder, C.l_hip, 264, 0;
    
    C.r_shoulder, C.thorax, 90, 1;
    C.thorax, C.l_shoulder, 90, 1;
    C.r_shoulder, C.l_shoulder, 180, 0;
    
    C.r_hip, C.pelvis, 66, 1;
    C.pelvis, C.l_hip, 66, 1;
    C.r_hip, C.l_hip, 132, 0;
    
    C.l_shoulder, C.thorax, 90, 1;
    C.thorax, C.r_shoulder, 90, 1;
    C.l_shoulder, C.r_shoulder, 180, 0;
    
    C.l_hip, C.pelvis, 66, 1;
    C.pelvis, C.r_hip, 66, 1;
    C.l_hip, C.r_hip, 132, 0;
    
    C.upper_neck, C.r_shoulder, 90, 1;
    C.upper_neck, C.l_shoulder, 90, 1;
    C.upper_neck, C.thorax, 60, 1;
    
    
    C.upper_neck, C.mid_body, 192, 0;
    C.mid_body, C.head_center, 252, 0;
    
    
    C.r_wrist, C.r_hand, 60, 1;
    
    C.l_wrist, C.l_hand, 60, 1;
    
    ];
C.CRF_pairs_ex4_r = C.CRF_pairs_ex4_r_b(:,1:3);
C.CRF_pairs_scl_r = [C.CRF_pairs_ex4_r(:,1:2),C.CRF_pairs_ex4_r(:,3)./1.92];

end
