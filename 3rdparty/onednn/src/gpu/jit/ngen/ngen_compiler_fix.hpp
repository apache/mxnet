/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 * Do not #include this file directly; ngen uses it internally.
 */

// This is a workaround for an ICC bug related to faulty
//  initialization of static constexpr LiteralType member variables
//  inside templated classes.
void _workaround_() {
    (void) r0.getBase(); (void) r1.getBase(); (void) r2.getBase(); (void) r3.getBase();
    (void) r4.getBase(); (void) r5.getBase(); (void) r6.getBase(); (void) r7.getBase();
    (void) r8.getBase(); (void) r9.getBase(); (void) r10.getBase(); (void) r11.getBase();
    (void) r12.getBase(); (void) r13.getBase(); (void) r14.getBase(); (void) r15.getBase();
    (void) r16.getBase(); (void) r17.getBase(); (void) r18.getBase(); (void) r19.getBase();
    (void) r20.getBase(); (void) r21.getBase(); (void) r22.getBase(); (void) r23.getBase();
    (void) r24.getBase(); (void) r25.getBase(); (void) r26.getBase(); (void) r27.getBase();
    (void) r28.getBase(); (void) r29.getBase(); (void) r30.getBase(); (void) r31.getBase();
    (void) r32.getBase(); (void) r33.getBase(); (void) r34.getBase(); (void) r35.getBase();
    (void) r36.getBase(); (void) r37.getBase(); (void) r38.getBase(); (void) r39.getBase();
    (void) r40.getBase(); (void) r41.getBase(); (void) r42.getBase(); (void) r43.getBase();
    (void) r44.getBase(); (void) r45.getBase(); (void) r46.getBase(); (void) r47.getBase();
    (void) r48.getBase(); (void) r49.getBase(); (void) r50.getBase(); (void) r51.getBase();
    (void) r52.getBase(); (void) r53.getBase(); (void) r54.getBase(); (void) r55.getBase();
    (void) r56.getBase(); (void) r57.getBase(); (void) r58.getBase(); (void) r59.getBase();
    (void) r60.getBase(); (void) r61.getBase(); (void) r62.getBase(); (void) r63.getBase();
    (void) r64.getBase(); (void) r65.getBase(); (void) r66.getBase(); (void) r67.getBase();
    (void) r68.getBase(); (void) r69.getBase(); (void) r70.getBase(); (void) r71.getBase();
    (void) r72.getBase(); (void) r73.getBase(); (void) r74.getBase(); (void) r75.getBase();
    (void) r76.getBase(); (void) r77.getBase(); (void) r78.getBase(); (void) r79.getBase();
    (void) r80.getBase(); (void) r81.getBase(); (void) r82.getBase(); (void) r83.getBase();
    (void) r84.getBase(); (void) r85.getBase(); (void) r86.getBase(); (void) r87.getBase();
    (void) r88.getBase(); (void) r89.getBase(); (void) r90.getBase(); (void) r91.getBase();
    (void) r92.getBase(); (void) r93.getBase(); (void) r94.getBase(); (void) r95.getBase();
    (void) r96.getBase(); (void) r97.getBase(); (void) r98.getBase(); (void) r99.getBase();
    (void) r100.getBase(); (void) r101.getBase(); (void) r102.getBase(); (void) r103.getBase();
    (void) r104.getBase(); (void) r105.getBase(); (void) r106.getBase(); (void) r107.getBase();
    (void) r108.getBase(); (void) r109.getBase(); (void) r110.getBase(); (void) r111.getBase();
    (void) r112.getBase(); (void) r113.getBase(); (void) r114.getBase(); (void) r115.getBase();
    (void) r116.getBase(); (void) r117.getBase(); (void) r118.getBase(); (void) r119.getBase();
    (void) r120.getBase(); (void) r121.getBase(); (void) r122.getBase(); (void) r123.getBase();
    (void) r124.getBase(); (void) r125.getBase(); (void) r126.getBase(); (void) r127.getBase();

    (void) null.getBase();
    (void) a0.getBase();

    (void) acc0.getBase(); (void) acc1.getBase(); (void) acc2.getBase(); (void) acc3.getBase();
    (void) acc4.getBase(); (void) acc5.getBase(); (void) acc6.getBase(); (void) acc7.getBase();
    (void) acc8.getBase(); (void) acc9.getBase();

    (void) mme0.getBase(); (void) mme1.getBase(); (void) mme2.getBase(); (void) mme3.getBase();
    (void) mme4.getBase(); (void) mme5.getBase(); (void) mme6.getBase(); (void) mme7.getBase();

    (void) noacc.getBase();
    (void) nomme.getBase();

    (void) f0.getBase();
    (void) f0_0.getBase();
    (void) f0_1.getBase();
    (void) f1.getBase();
    (void) f1_0.getBase();
    (void) f1_1.getBase();

    (void) ce0.getBase();
    (void) sp.getBase();
    (void) sr0.getBase();
    (void) sr1.getBase();
    (void) cr0.getBase();
    (void) n0.getBase();
    (void) ip.getBase();
    (void) tdr0.getBase();
    (void) tm0.getBase();
    (void) pm0.getBase();
    (void) tp0.getBase();
    (void) dbg0.getBase();

    (void) NoDDClr.getAll();
    (void) NoDDChk.getAll();
    (void) AccWrEn.getAll();
    (void) NoSrcDepSet.getAll();
    (void) Breakpoint.getAll();
    (void) sat.getAll();
    (void) NoMask.getAll();
    (void) Serialize.getAll();
    (void) EOT.getAll();
    (void) Align1.getAll();
    (void) Align16.getAll();
    (void) Atomic.getAll();
    (void) Switch.getAll();
    (void) NoPreempt.getAll();

    (void) x_repl.getAll();
    (void) y_repl.getAll();
    (void) z_repl.getAll();
    (void) w_repl.getAll();

    (void) ze.getAll();
    (void) eq.getAll();
    (void) nz.getAll();
    (void) ne.getAll();
    (void) gt.getAll();
    (void) ge.getAll();
    (void) lt.getAll();
    (void) le.getAll();
    (void) ov.getAll();
    (void) un.getAll();
    (void) eo.getAll();

    (void) M0.getAll();
    (void) M4.getAll();
    (void) M8.getAll();
    (void) M12.getAll();
    (void) M16.getAll();
    (void) M20.getAll();
    (void) M24.getAll();
    (void) M28.getAll();

    (void) SBInfo(sb0).getID();  (void) SBInfo(sb1).getID();  (void) SBInfo(sb2).getID();  (void) SBInfo(sb3).getID();
    (void) SBInfo(sb4).getID();  (void) SBInfo(sb5).getID();  (void) SBInfo(sb6).getID();  (void) SBInfo(sb7).getID();
    (void) SBInfo(sb8).getID();  (void) SBInfo(sb9).getID();  (void) SBInfo(sb10).getID(); (void) SBInfo(sb11).getID();
    (void) SBInfo(sb12).getID(); (void) SBInfo(sb13).getID(); (void) SBInfo(sb14).getID(); (void) SBInfo(sb15).getID();

    (void) A32.getModel();
    (void) A32NC.getModel();
    (void) A64.getModel();
    (void) A64NC.getModel();
    (void) SLM.getModel();
}
