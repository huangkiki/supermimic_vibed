from copy import deepcopy
import unittest

from experiments.ablation.analyze_policy_contacts import checked_contacts, contact_evidence


class PolicyContactTests(unittest.TestCase):
    def fixture(self):
        contact = ((0, 1),)
        bank = dict(source_sample_indices=[7], motion_names=['walk'], prefix_actions=[0, 0, 0],
                    prefix_contacts=[[contact], [contact], [contact]])
        record = dict(geometry_contacts=[[contact], [contact], [()], [()]])
        rollout = dict(baselines=[record, deepcopy(record)], actual_adam=deepcopy(record),
                       perturbations={'random_0': [(deepcopy(record), deepcopy(record), .01, .01)]})
        return bank, rollout

    def test_past_initial_and_future_labels_are_distinct(self):
        bank, rollout = self.fixture()
        result = contact_evidence(bank, rollout, [0, 1], group='stable_contact', indices=[7], motions=['walk'], horizon=4)
        self.assertEqual(result[0]['past_contact_group'], 'stable_contact')
        # Initial s0 is the LAST prefix state, not the first post-transition state.
        self.assertEqual(result[0]['initial_symmetric_group'], 'stable_contact')
        self.assertEqual(result[0]['future_window_group'], 'switch_nearby')
        self.assertEqual(result[0]['future_contact_switches'], 1)

    def test_missing_contacts_in_any_perturbed_rollout_are_rejected(self):
        bank, rollout = self.fixture()
        rollout['perturbations']['random_0'][0][1].pop('geometry_contacts')
        with self.assertRaisesRegex(ValueError, 'contact layout'):
            contact_evidence(bank, rollout, [0, 1], group='stable_contact', indices=[7], motions=['walk'], horizon=4)

    def test_changed_selection_or_past_label_is_rejected(self):
        for field, value in [('source_sample_indices', [8]), ('motion_names', ['run']),
                             ('prefix_contacts', [[()], [()], [()]])]:
            with self.subTest(field=field):
                bank, rollout = self.fixture()
                bank[field] = value
                with self.assertRaises(ValueError):
                    contact_evidence(bank, rollout, [0, 1], group='stable_contact', indices=[7], motions=['walk'], horizon=4)

    def test_bad_contact_layout_or_geometry_map_is_rejected(self):
        for contacts, mapping in [([[]], [0, 1]), ([[((0, 3),)]], [0, 1]), ([[((0, 1),)]], [-1, 2])]:
            with self.subTest(contacts=contacts, mapping=mapping), self.assertRaises(ValueError):
                checked_contacts(contacts, 1, 1, mapping)


if __name__ == '__main__':
    unittest.main()
