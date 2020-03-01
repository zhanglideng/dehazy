import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def write_excel(sheet, data_type, line, epoch, itr, loss):
    # 在train页中保留train的数据，在validation页中保留validation的数据
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间]
    """
    :param sheet:
    :param data_type:
    :param line:
    :param epoch:
    :param itr:
    :param loss:
    :return:
    一共有11个loss
    train=["EPOCH", "ITR", "L2_LOSS", "SSIM_LOSS", "LOSS", "LR"]
    val=["EPOCH", "L2_LOSS", "SSIM_LOSS", "LOSS", "PSNR", "SSIM", "LR"]
    """
    sum_loss = 0
    if data_type == 'train':
        sheet.write(line, 0, epoch + 1)
        sheet.write(line, 1, itr + 1)
        for i in range(12):
            sheet.write(line, i + 2, round(loss[i], 4))
            sum_loss += loss[i]
        sheet.write(line, 14, round(sum_loss, 4))
    else:
        if len(loss) > 1:
            sheet.write(line, 0, epoch + 1)
            for i in range(12):
                sheet.write(line, i + 1, round(loss[i], 4))
                sum_loss += loss[i]
            sheet.write(line, 13, round(sum_loss, 4))
        else:
            sheet.write(line, 0, epoch + 1)
            sheet.write(line, 1, round(loss, 4))
    return line + 1


def init_excel():
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('train', cell_overwrite_ok=True)
    sheet2 = workbook.add_sheet('val', cell_overwrite_ok=True)
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
    row0 = ["EPOCH", "ITR", "At_J_l2", "At_J_ssim", "At_A_l2", "At_t_l2", "At_t_ssim",
            "ed_re_l2", "ed_re_ssim", "ed_dehaze_l2", "ed_dehaze_ssim", "ed_sc_l2",
            "ed_I_l2", "ed_I_ssim", "loss"]
    row1 = ["EPOCH", "At_J_l2", "At_J_ssim", "At_A_l2", "At_t_l2", "At_t_ssim",
            "ed_re_l2", "ed_re_ssim", "ed_dehaze_l2", "ed_dehaze_ssim", "ed_sc_l2",
            "ed_I_l2", "ed_I_ssim", "loss"]
    for i in range(0, len(row0)):
        print('写入train_excel')
        sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
    for i in range(0, len(row1)):
        print('写入val_excel')
        sheet2.write(0, i, row1[i], set_style('Times New Roman', 220, True))
    return workbook, sheet1, sheet2
