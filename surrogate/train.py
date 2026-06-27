"""Surrogate training loop."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from tqdm import tqdm

from common.paths import ATHLETES_183_RETARGETED_DIR
from datasets.athletes_retarget import retargeted183_data_loader
from surrogate.model import MLPModel
from surrogate.mot_io import write_muscle_activations


def main() -> None:
    window_size = 64
    batch_size = 16
    train_loader = retargeted183_data_loader(
        window_size=window_size,
        unit_length=4,
        batch_size=batch_size,
        num_workers=0,
        data_dir=ATHLETES_183_RETARGETED_DIR,
        pre_load=False,
    )

    sample_batch = next(iter(train_loader))
    motion_sample = sample_batch[0]
    _, in_t, in_d = motion_sample.shape
    out_t, out_d = in_t, 80

    input_dim = in_t * in_d
    hidden_dim = max(1024, in_t * in_d // 4)
    output_dim = out_t * out_d
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = MLPModel(input_dim, hidden_dim, output_dim).to(device)
    print("Model:", model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1, weight_decay=1e-2)

    best_test_loss = float("inf")
    best_test_loss_epoch = -1
    best_model = None
    restore_cnt = 0
    current_lr = 1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            inputs = inputs.float().to(device)
            outputs = model(inputs)
            targets = inputs
            loss_main = criterion(outputs, targets)
            loss_temporal = criterion(outputs[:, 1:], outputs[:, :-1])
            loss = loss_main + loss_temporal
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        test_loss = 0.0
        test_l = 0.0
        avg_l_per_timestep = torch.zeros(out_t).to(device)
        avg_cum_thigh_activation = 0.0
        avg_cum_thigh_activation_pred = 0.0

        test_loader = retargeted183_data_loader(
            window_size=window_size,
            unit_length=4,
            batch_size=batch_size,
            num_workers=0,
            data_dir=ATHLETES_183_RETARGETED_DIR,
            pre_load=False,
        )

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].float().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
                if out_d > 4:
                    l = criterion(outputs[:, :, : out_d - 4], inputs[:, :, : out_d - 4])
                    test_l += l.item()
                    avg_cum_thigh_activation += torch.sum(inputs[:, :, : out_d - 4]) / 4
                    avg_cum_thigh_activation_pred += torch.sum(outputs[:, :, : out_d - 4]) / 4
                    avg_l_per_timestep += torch.sum(
                        (outputs[:, :, : out_d - 4] - inputs[:, :, : out_d - 4]) ** 2, dim=0
                    ).mean(dim=-1)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_l = test_l / len(test_loader)
        avg_cum_thigh_activation = avg_cum_thigh_activation / len(test_loader)
        avg_cum_thigh_activation_pred = avg_cum_thigh_activation_pred / len(test_loader)
        avg_l_per_timestep = torch.sqrt(avg_l_per_timestep / len(test_loader))
        print(
            f"Epoch {epoch+1}, Best model:{best_test_loss_epoch} "
            f"Test Loss: {avg_test_loss:.6f} Norm:{model.fc_main.weight.norm()} "
            f"Thigh loss:{avg_test_l:.6f}"
        )

        if epoch - best_test_loss_epoch > 20:
            if avg_test_loss > best_test_loss:
                model.load_state_dict(best_model)
                print(
                    "Model restored at epoch:", epoch,
                    " to best model at epoch:", best_test_loss_epoch,
                )
                restore_cnt += 1
                if restore_cnt > 10:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.1
                        print("Learning rate reduced to 1/10th of its value")
                        current_lr = min(current_lr, param_group["lr"])
                    if current_lr < 1e-5:
                        print(f"Learning rate is less than 1e-5. Exiting:{current_lr}")
                        break
                    restore_cnt = 0
            else:
                best_model = model.state_dict()
                best_test_loss_epoch = epoch
                torch.save(model.state_dict(), "surrogate_model.pth")
                print(
                    "Model saved at epoch:", best_test_loss_epoch,
                    " Loss:", avg_test_loss, " Prev loss:", best_test_loss,
                )
                best_test_loss = avg_test_loss

            print(
                f"Expected Thigh activation:{avg_cum_thigh_activation} "
                f"Predicted:{avg_cum_thigh_activation_pred}"
            )
            print(f"RMSE per timestep:{avg_l_per_timestep}")

    final_test_loader = retargeted183_data_loader(
        window_size=window_size,
        unit_length=4,
        batch_size=1,
        num_workers=0,
        data_dir=ATHLETES_183_RETARGETED_DIR,
        pre_load=False,
    )

    collate_predictions = {}
    for batch in final_test_loader:
        try:
            inputs = batch[0].float().to(device)
            names = batch[2]
            subject_ids = batch[3]
            outputs = model(inputs)
            name = names[0]
            subject_id = subject_ids[0]
            pred = outputs[0].detach().cpu().numpy()
            collate_predictions[name] = (subject_id, pred)
        except Exception as e:
            print("Error: processing final output", e)

    save_dir = os.path.join(ATHLETES_183_RETARGETED_DIR, "surrogate_activations")
    os.makedirs(save_dir, exist_ok=True)
    for name, (subject_id, arr) in collate_predictions.items():
        act_name = f"{subject_id}-{name}.mot"
        print("Saving to", os.path.join(save_dir, act_name))
        write_muscle_activations(os.path.join(save_dir, act_name), arr)


if __name__ == "__main__":
    main()
