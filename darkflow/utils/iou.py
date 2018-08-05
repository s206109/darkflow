import numpy as np

def adjust_box(Anbox, Prbox ):
    xamin = Anbox[1]
    yamin = Anbox[2]
    xamax = Anbox[3]
    yamax = Anbox[4]

    xbmin = Prbox[1]
    ybmin = Prbox[2]
    xbmax = Prbox[3]
    ybmax = Prbox[4]

    w_a = xamax - xamin
    w_b = xbmax - xbmin
    h_a = yamax - yamin
    h_b = ybmax - ybmin

    centerx_a = xamax - w_a * 0.5
    centerx_b = xbmax - w_b * 0.5
    centery_a = yamax - h_a * 0.5
    centery_b = ybmax - h_b * 0.5
    import pdb; pdb.set_trace()
    return box_iou(w_a ,w_b, h_a, h_b, centerx_a, centery_a, centerx_b, centery_b)



def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    import pdb; pdb.set_trace()
    return right - left;

def box_intersection(wa, wb, ha, hb, cax, cay, cbx, cby):
    w = overlap(cax, wa, cbx, wb);
    h = overlap(cay, ha, cby, hb);
    import pdb; pdb.set_trace()
    if w < 0 or h < 0: return 0;
    area = w * h;
    import pdb; pdb.set_trace()
    return area;

def box_union(wa, wb, ha, hb, cax, cay, cbx, cby):
    i = box_intersection(wa, wb, ha, hb, cax, cay, cbx, cby);
    u = wa * ha + wb * hb - i;
    return u;

def box_iou(wa, wb, ha, hb, cax, cay, cbx, cby):
    return box_intersection(wa, wb, ha, hb, cax, cay, cbx, cby) / box_union(wa, wb, ha, hb, cax, cay, cbx, cby);

def prob_compare(box):
    return box.probs[box.class_num]

def prob_compare2(boxa, boxb):
    if (boxa.pi < boxb.pi):
        return 1
    elif(boxa.pi == boxb.pi):
        return 0
    else:
        return -1
